// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"

#include "4C_comm_utils_gid_vector.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_coupling_volmortar_shape.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_dofset_predefineddofnumber.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fluid_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_mortar_projector.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_boundary_calc.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i_elch.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::MeshtyingStrategyS2I::MeshtyingStrategyS2I(
    ScaTra::ScaTraTimIntImpl* scatratimint, const Teuchos::ParameterList& parameters)
    : MeshtyingStrategyBase(scatratimint),
      interfacemaps_(nullptr),
      blockmaps_slave_(nullptr),
      blockmaps_master_(nullptr),
      icoup_(nullptr),
      icoupmortar_(),
      imortarcells_(),
      imortarredistribution_(
          Teuchos::getIntegralValue<Inpar::S2I::CouplingType>(parameters.sublist("S2I COUPLING"),
              "COUPLINGTYPE") == Inpar::S2I::coupling_mortar_standard and
          Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
              Global::Problem::instance()->mortar_coupling_params().sublist(
                  "PARALLEL REDISTRIBUTION"),
              "PARALLEL_REDIST") != Inpar::Mortar::ParallelRedist::redist_none),
      islavemap_(nullptr),
      imastermap_(nullptr),
      islavenodestomasterelements_(),
      islavenodesimpltypes_(),
      islavenodeslumpedareas_(),
      islavematrix_(nullptr),
      imastermatrix_(nullptr),
      imasterslavematrix_(nullptr),
      couplingtype_(Teuchos::getIntegralValue<Inpar::S2I::CouplingType>(
          parameters.sublist("S2I COUPLING"), "COUPLINGTYPE")),
      D_(nullptr),
      M_(nullptr),
      E_(nullptr),
      P_(nullptr),
      Q_(nullptr),
      lm_(nullptr),
      extendedmaps_(nullptr),
      lmresidual_(nullptr),
      lmincrement_(nullptr),
      islavetomastercoltransform_(nullptr),
      islavetomasterrowtransform_(nullptr),
      islavetomasterrowcoltransform_(nullptr),
      islaveresidual_(nullptr),
      imasterresidual_(nullptr),
      islavephidtnp_(nullptr),
      imasterphidt_on_slave_side_np_(nullptr),
      imasterphi_on_slave_side_np_(nullptr),
      lmside_(Teuchos::getIntegralValue<Inpar::S2I::InterfaceSides>(
          parameters.sublist("S2I COUPLING"), "LMSIDE")),
      matrixtype_(Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(parameters, "MATRIXTYPE")),
      ntsprojtol_(parameters.sublist("S2I COUPLING").get<double>("NTSPROJTOL")),
      intlayergrowth_evaluation_(Teuchos::getIntegralValue<Inpar::S2I::GrowthEvaluation>(
          parameters.sublist("S2I COUPLING"), "INTLAYERGROWTH_EVALUATION")),
      intlayergrowth_convtol_(
          parameters.sublist("S2I COUPLING").get<double>("INTLAYERGROWTH_CONVTOL")),
      intlayergrowth_itemax_(parameters.sublist("S2I COUPLING").get<int>("INTLAYERGROWTH_ITEMAX")),
      intlayergrowth_timestep_(
          parameters.sublist("S2I COUPLING").get<double>("INTLAYERGROWTH_TIMESTEP")),
      blockmapgrowth_(nullptr),
      extendedblockmaps_(nullptr),
      extendedsystemmatrix_(nullptr),
      extendedsolver_(nullptr),
      growthn_(nullptr),
      growthnp_(nullptr),
      growthdtn_(nullptr),
      growthdtnp_(nullptr),
      growthhist_(nullptr),
      growthresidual_(nullptr),
      growthincrement_(nullptr),
      scatragrowthblock_(nullptr),
      growthscatrablock_(nullptr),
      growthgrowthblock_(nullptr),
      equilibration_(nullptr),
      output_interface_flux_(parameters.sublist("S2I COUPLING").get<bool>("OUTPUT_INTERFACE_FLUX")),
      has_capacitive_contributions_(false),
      kinetics_conditions_meshtying_slaveside_(),
      slaveonly_(parameters.sublist("S2I COUPLING").get<bool>("SLAVEONLY")),
      independent_setup_of_conditions_(
          parameters.sublist("S2I COUPLING").get<bool>("MESHTYING_CONDITIONS_INDEPENDENT_SETUP"))
{
  // empty constructor
}  // ScaTra::meshtying_strategy_s2_i::meshtying_strategy_s2_i


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::condense_mat_and_rhs(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    const bool calcinittimederiv) const
{
  switch (couplingtype_)
  {
    case Inpar::S2I::coupling_mortar_condensed_bubnov:
    {
      // extract global system matrix
      std::shared_ptr<Core::LinAlg::SparseMatrix> sparsematrix = scatratimint_->system_matrix();
      if (sparsematrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

      if (lmside_ == Inpar::S2I::side_slave)
      {
        // initialize temporary matrix for slave-side rows of global system matrix
        Core::LinAlg::SparseMatrix sparsematrixrowsslave(*interfacemaps_->map(1), 81);

        // extract slave-side rows of global system matrix into temporary matrix
        extract_matrix_rows(*sparsematrix, sparsematrixrowsslave, *interfacemaps_->map(1));

        // finalize temporary matrix with slave-side rows of global system matrix
        sparsematrixrowsslave.complete(*interfacemaps_->full_map(), *interfacemaps_->map(1));

        // zero out slave-side rows of global system matrix after having extracted them into
        // temporary matrix
        sparsematrix->complete();
        sparsematrix->apply_dirichlet(*interfacemaps_->map(1), false);

        // apply scatra-scatra interface coupling
        if (not slaveonly_)
        {
          // replace slave-side rows of global system matrix by projected slave-side rows including
          // interface contributions
          Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                       *Q_, true, sparsematrixrowsslave, false, false, false, true),
              false, 1., *sparsematrix, 1.);
          // during calculation of initial time derivative, standard global system matrix is
          // replaced by global mass matrix, and hence interface contributions must not be included
          if (!calcinittimederiv)
            Core::LinAlg::matrix_add(*islavematrix_, false, 1., *sparsematrix, 1.);
        }

        // apply standard meshtying
        else
        {
          Core::LinAlg::matrix_add(*D_, false, 1., *sparsematrix, 1.);
          Core::LinAlg::matrix_add(*M_, false, -1., *sparsematrix, 1.);
        }

        // add projected slave-side rows to master-side rows of global system matrix
        Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                     *P_, true, sparsematrixrowsslave, false, false, false, true),
            false, 1., *sparsematrix, 1.);

        // extract slave-side entries of global residual vector
        std::shared_ptr<Core::LinAlg::Vector<double>> residualslave =
            interfacemaps_->extract_vector(*scatratimint_->residual(), 1);

        // apply scatra-scatra interface coupling
        if (not slaveonly_)
        {
          // replace slave-side entries of global residual vector by projected slave-side entries
          // including interface contributions
          Core::LinAlg::Vector<double> Q_residualslave(*interfacemaps_->map(1));
          Q_->multiply(true, *residualslave, Q_residualslave);
          interfacemaps_->insert_vector(Q_residualslave, 1, *scatratimint_->residual());
          interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
        }

        // apply standard meshtying
        else
        {
          // zero out slave-side entries of global residual vector
          interfacemaps_->put_scalar(*scatratimint_->residual(), 1, 0.);
        }

        // add projected slave-side entries to master-side entries of global residual vector
        Core::LinAlg::Vector<double> P_residualslave(*interfacemaps_->map(2));
        P_->multiply(true, *residualslave, P_residualslave);
        interfacemaps_->add_vector(P_residualslave, 2, *scatratimint_->residual());
      }

      else
      {
        // initialize temporary matrix for master-side rows of global system matrix
        Core::LinAlg::SparseMatrix sparsematrixrowsmaster(*interfacemaps_->map(2), 81);

        // extract master-side rows of global system matrix into temporary matrix
        extract_matrix_rows(*sparsematrix, sparsematrixrowsmaster, *interfacemaps_->map(2));

        // finalize temporary matrix with master-side rows of global system matrix
        sparsematrixrowsmaster.complete(*interfacemaps_->full_map(), *interfacemaps_->map(2));

        // zero out master-side rows of global system matrix after having extracted them into
        // temporary matrix and replace them by projected master-side rows including interface
        // contributions
        sparsematrix->complete();
        sparsematrix->apply_dirichlet(*interfacemaps_->map(2), false);
        Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                     *Q_, true, sparsematrixrowsmaster, false, false, false, true),
            false, 1., *sparsematrix, 1.);
        // during calculation of initial time derivative, standard global system matrix is replaced
        // by global mass matrix, and hence interface contributions must not be included
        if (!calcinittimederiv)
          Core::LinAlg::matrix_add(*imastermatrix_, false, 1., *sparsematrix, 1.);

        // add projected master-side rows to slave-side rows of global system matrix
        Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                     *P_, true, sparsematrixrowsmaster, false, false, false, true),
            false, 1., *sparsematrix, 1.);

        // extract master-side entries of global residual vector
        std::shared_ptr<Core::LinAlg::Vector<double>> residualmaster =
            interfacemaps_->extract_vector(*scatratimint_->residual(), 2);

        // replace master-side entries of global residual vector by projected master-side entries
        // including interface contributions
        Core::LinAlg::Vector<double> Q_residualmaster(*interfacemaps_->map(2));
        Q_->multiply(true, *residualmaster, Q_residualmaster);
        interfacemaps_->insert_vector(Q_residualmaster, 2, *scatratimint_->residual());
        interfacemaps_->add_vector(
            imasterresidual_->as_multi_vector(), 2, *scatratimint_->residual());

        // add projected master-side entries to slave-side entries of global residual vector
        Core::LinAlg::Vector<double> P_residualmaster(*interfacemaps_->map(1));
        P_->multiply(true, *residualmaster, P_residualmaster);
        interfacemaps_->add_vector(P_residualmaster, 1, *scatratimint_->residual());
      }

      break;
    }

    case Inpar::S2I::coupling_matching_nodes:
    case Inpar::S2I::coupling_mortar_standard:
    case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
    case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
    case Inpar::S2I::coupling_mortar_condensed_petrov:
    case Inpar::S2I::coupling_nts_standard:
    {
      // do nothing in these cases
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of mortar meshtying for scatra-scatra interface coupling not recognized!");
      break;
    }
  }
}


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
const Core::LinAlg::Map& ScaTra::MeshtyingStrategyS2I::dof_row_map() const
{
  return extendedmaps_ != nullptr ? *extendedmaps_->full_map() : *scatratimint_->dof_row_map();
}


/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_meshtying()
{
  // time measurement: evaluate condition 'S2IMeshtying'
  TEUCHOS_FUNC_TIME_MONITOR("SCATRA:       + evaluate condition 'S2IMeshtying'");

  switch (couplingtype_)
  {
    case Inpar::S2I::coupling_matching_nodes:
    {
      // create parameter list
      Teuchos::ParameterList condparams;

      // action for elements
      Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
          "action", ScaTra::BoundaryAction::calc_s2icoupling, condparams);

      // set global state vectors according to time-integration scheme
      scatratimint_->add_time_integration_specific_vectors();

      // evaluate scatra-scatra interface coupling at time t_{n+1} or t_{n+alpha_F}
      islavematrix_->zero();
      if (not slaveonly_) imastermatrix_->zero();
      islaveresidual_->put_scalar(0.);
      for (auto kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
      {
        if (kinetics_slave_cond.second->parameters().get<Inpar::S2I::KineticModels>(
                "KINETIC_MODEL") != static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux) and
            kinetics_slave_cond.second->g_type() != Core::Conditions::geometry_type_point)
        {
          // collect condition specific data and store to scatra boundary parameter class
          set_condition_specific_scatra_parameters(*kinetics_slave_cond.second);

          if (not slaveonly_)
          {
            scatratimint_->discretization()->evaluate_condition(condparams, islavematrix_,
                imastermatrix_, islaveresidual_, nullptr, nullptr, "S2IKinetics",
                kinetics_slave_cond.second->parameters().get<int>("ConditionID"));
          }
          else
          {
            scatratimint_->discretization()->evaluate_condition(condparams, islavematrix_, nullptr,
                islaveresidual_, nullptr, nullptr, "S2IKinetics",
                kinetics_slave_cond.second->parameters().get<int>("ConditionID"));
          }
        }
      }

      // finalize interface matrices
      islavematrix_->complete();
      if (not slaveonly_) imastermatrix_->complete();

      // assemble global system matrix depending on matrix type
      switch (matrixtype_)
      {
        case Core::LinAlg::MatrixType::sparse:
        {
          // check matrix
          std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix = scatratimint_->system_matrix();
          FOUR_C_ASSERT(systemmatrix != nullptr, "System matrix is not a sparse matrix!");

          // assemble linearizations of slave fluxes w.r.t. slave dofs into global system matrix
          Core::LinAlg::matrix_add(*islavematrix_, false, 1., *systemmatrix, 1.);

          if (not slaveonly_)
          {
            // transform linearizations of slave fluxes w.r.t. master dofs and assemble into global
            // system matrix
            (*islavetomastercoltransform_)(imastermatrix_->row_map(), imastermatrix_->col_map(),
                *imastermatrix_, 1., Coupling::Adapter::CouplingSlaveConverter(*icoup_),
                *systemmatrix, true, true);

            // derive linearizations of master fluxes w.r.t. slave dofs and assemble into global
            // system matrix
            (*islavetomasterrowtransform_)(*islavematrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true);

            // derive linearizations of master fluxes w.r.t. master dofs and assemble into global
            // system matrix
            (*islavetomasterrowcoltransform_)(*imastermatrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_),
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true, true);
          }

          // In case the interface linearizations and residuals are evaluated on slave side only,
          // we now apply a standard meshtying algorithm to condense out the slave-side degrees of
          // freedom.
          else if (!scatratimint_->discretization()->has_condition("PointCoupling"))
          {
            // initialize temporary matrix for slave-side rows of system matrix
            Core::LinAlg::SparseMatrix systemmatrixrowsslave(*icoup_->slave_dof_map(), 81);

            // extract slave-side rows of system matrix into temporary matrix
            extract_matrix_rows(*systemmatrix, systemmatrixrowsslave, *icoup_->slave_dof_map());

            // zero out slave-side rows of system matrix and put a one on the main diagonal
            systemmatrix->complete();
            systemmatrix->apply_dirichlet(*icoup_->slave_dof_map(), true);
            systemmatrix->un_complete();

            // loop over all slave-side rows of system matrix
            for (int slavedoflid = 0; slavedoflid < icoup_->slave_dof_map()->num_my_elements();
                ++slavedoflid)
            {
              // determine global ID of current matrix row
              const int slavedofgid = icoup_->slave_dof_map()->gid(slavedoflid);
              if (slavedofgid < 0) FOUR_C_THROW("Couldn't find local ID {} in map!", slavedoflid);

              // determine global ID of associated master-side matrix column
              const int masterdofgid = icoup_->perm_master_dof_map()->gid(slavedoflid);
              if (masterdofgid < 0)
                FOUR_C_THROW("Couldn't find local ID {} in permuted map!", slavedoflid);

              // insert value -1. into intersection of slave-side row and master-side column in
              // system matrix this effectively forces the slave-side degree of freedom to assume
              // the same value as the master-side degree of freedom
              const double value(-1.);
              systemmatrix->insert_global_values(slavedofgid, 1, &value, &masterdofgid);

              // insert zero into intersection of slave-side row and master-side column in temporary
              // matrix this prevents the system matrix from changing its graph when calling this
              // function again during the next Newton iteration
              const double zero(0.);
              systemmatrixrowsslave.insert_global_values(slavedofgid, 1, &zero, &masterdofgid);
            }

            // finalize temporary matrix with slave-side rows of system matrix
            systemmatrixrowsslave.complete(*scatratimint_->dof_row_map(), *icoup_->slave_dof_map());

            // add slave-side rows of system matrix to corresponding master-side rows to finalize
            // matrix condensation of slave-side degrees of freedom
            (*islavetomasterrowtransform_)(systemmatrixrowsslave, 1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true);
          }
          break;
        }
        case Core::LinAlg::MatrixType::block_condition:
        case Core::LinAlg::MatrixType::block_condition_dof:
        {
          // check matrix
          std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksystemmatrix =
              scatratimint_->block_system_matrix();
          FOUR_C_ASSERT(blocksystemmatrix != nullptr, "System matrix is not a block matrix!");

          std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkss(
              Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                  *islavematrix_, *blockmaps_slave_, *blockmaps_slave_));
          blockkss->complete();

          // assemble interface block matrix into global block system matrix
          blocksystemmatrix->add(*blockkss, false, 1., 1.);

          if (not slaveonly_)
          {
            std::shared_ptr<Core::LinAlg::SparseMatrix> ksm(
                std::make_shared<Core::LinAlg::SparseMatrix>(*icoup_->slave_dof_map(), 81, false));
            std::shared_ptr<Core::LinAlg::SparseMatrix> kms(
                std::make_shared<Core::LinAlg::SparseMatrix>(*icoup_->master_dof_map(), 81, false));
            std::shared_ptr<Core::LinAlg::SparseMatrix> kmm(
                std::make_shared<Core::LinAlg::SparseMatrix>(*icoup_->master_dof_map(), 81, false));

            // transform linearizations of slave fluxes w.r.t. master dofs
            (*islavetomastercoltransform_)(imastermatrix_->row_map(), imastermatrix_->col_map(),
                *imastermatrix_, 1., Coupling::Adapter::CouplingSlaveConverter(*icoup_), *ksm);
            ksm->complete(*icoup_->master_dof_map(), *icoup_->slave_dof_map());

            // derive linearizations of master fluxes w.r.t. slave dofs
            (*islavetomasterrowtransform_)(
                *islavematrix_, -1., Coupling::Adapter::CouplingSlaveConverter(*icoup_), *kms);
            kms->complete(*icoup_->slave_dof_map(), *icoup_->master_dof_map());

            // derive linearizations of master fluxes w.r.t. master dofs
            (*islavetomasterrowcoltransform_)(*imastermatrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_),
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *kmm);
            kmm->complete();

            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockksm(
                Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                    *ksm, *blockmaps_master_, *blockmaps_slave_));
            blockksm->complete();
            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkms(
                Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                    *kms, *blockmaps_slave_, *blockmaps_master_));
            blockkms->complete();
            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkmm(
                Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                    *kmm, *blockmaps_master_, *blockmaps_master_));
            blockkmm->complete();

            // assemble interface block matrices into global block system matrix
            blocksystemmatrix->add(*blockksm, false, 1., 1.);
            blocksystemmatrix->add(*blockkms, false, 1., 1.);
            blocksystemmatrix->add(*blockkmm, false, 1., 1.);
          }

          // safety check
          else
          {
            FOUR_C_THROW(
                "Scatra-scatra interface coupling with evaluation of interface linearizations and "
                "residuals on slave side only is not yet available for block system matrices!");
          }

          break;
        }

        default:
        {
          FOUR_C_THROW(
              "Type of global system matrix for scatra-scatra interface coupling not recognized!");
          break;
        }
      }

      // assemble slave residuals into global residual vector
      interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());

      if (not slaveonly_)
      {
        // transform master residuals and assemble into global residual vector
        interfacemaps_->add_vector(
            *icoup_->slave_to_master(*islaveresidual_), 2, *scatratimint_->residual(), -1.);
      }
      // In case the interface linearizations and residuals are evaluated on slave side only,
      // we now apply a standard meshtying algorithm to condense out the slave-side degrees of
      // freedom.
      else if (!scatratimint_->discretization()->has_condition("PointCoupling"))
      {
        // initialize temporary vector for slave-side entries of residual vector
        Core::LinAlg::Vector<double> residualslave(*icoup_->slave_dof_map());

        // loop over all slave-side entries of residual vector
        for (int slavedoflid = 0; slavedoflid < icoup_->slave_dof_map()->num_my_elements();
            ++slavedoflid)
        {
          // determine global ID of current vector entry
          const int slavedofgid = icoup_->slave_dof_map()->gid(slavedoflid);
          if (slavedofgid < 0) FOUR_C_THROW("Couldn't find local ID {} in map!", slavedoflid);

          // copy current vector entry into temporary vector
          residualslave.replace_global_value(slavedofgid,
              scatratimint_->residual()
                  ->local_values_as_span()[scatratimint_->dof_row_map()->lid(slavedofgid)]);

          // zero out current vector entry
          scatratimint_->residual()->replace_global_value(slavedofgid, 0.);
        }

        // add slave-side entries of residual vector to corresponding master-side entries to
        // finalize vector condensation of slave-side degrees of freedom
        interfacemaps_->add_vector(
            *icoup_->slave_to_master(residualslave), 2, *scatratimint_->residual());
      }

      if (has_capacitive_contributions_) evaluate_and_assemble_capacitive_contributions();

      break;
    }

    case Inpar::S2I::coupling_mortar_standard:
    case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
    case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
    case Inpar::S2I::coupling_mortar_condensed_petrov:
    case Inpar::S2I::coupling_mortar_condensed_bubnov:
    case Inpar::S2I::coupling_nts_standard:
    {
      // initialize auxiliary system matrix and vector for slave side
      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_slave or couplingtype_ == Inpar::S2I::coupling_nts_standard)
      {
        islavematrix_->zero();
        islaveresidual_->put_scalar(0.);
      }

      // initialize auxiliary system matrix and vector for master side
      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_master or couplingtype_ == Inpar::S2I::coupling_nts_standard)
      {
        imastermatrix_->zero();
        imasterresidual_->put_scalar(0.);
      }

      // loop over all scatra-scatra coupling interfaces
      for (auto& kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
      {
        // extract mortar interface discretization
        Core::FE::Discretization& idiscret =
            icoupmortar_[kinetics_slave_cond.first]->interface()->discret();

        // export global state vector to mortar interface
        Core::LinAlg::Vector<double> iphinp(*idiscret.dof_col_map(), false);
        Core::LinAlg::export_to(*scatratimint_->phiafnp(), iphinp);
        idiscret.set_state("iphinp", iphinp);

        // create parameter list for mortar integration cells
        Teuchos::ParameterList params;

        // add current condition to parameter list
        params.set<const Core::Conditions::Condition*>("condition", kinetics_slave_cond.second);

        // collect condition specific data and store to scatra boundary parameter class
        set_condition_specific_scatra_parameters(*(kinetics_slave_cond.second));

        if (couplingtype_ != Inpar::S2I::coupling_nts_standard)
        {
          // set action
          params.set<Inpar::S2I::EvaluationActions>("action", Inpar::S2I::evaluate_condition);

          evaluate_mortar_cells(idiscret, params, islavematrix_, Inpar::S2I::side_slave,
              Inpar::S2I::side_slave, islavematrix_, Inpar::S2I::side_slave,
              Inpar::S2I::side_master, imastermatrix_, Inpar::S2I::side_master,
              Inpar::S2I::side_slave, imastermatrix_, Inpar::S2I::side_master,
              Inpar::S2I::side_master,
              islaveresidual_ != nullptr
                  ? Core::Utils::shared_ptr_from_ref(islaveresidual_->as_multi_vector())
                  : nullptr,
              Inpar::S2I::side_slave, imasterresidual_, Inpar::S2I::side_master);
        }

        else
        {
          // set action
          params.set<Inpar::S2I::EvaluationActions>("action", Inpar::S2I::evaluate_condition_nts);

          // evaluate note-to-segment coupling at current interface
          evaluate_nts(*islavenodestomasterelements_[kinetics_slave_cond.first],
              *islavenodeslumpedareas_[kinetics_slave_cond.first],
              *islavenodesimpltypes_[kinetics_slave_cond.first], idiscret, params, islavematrix_,
              Inpar::S2I::side_slave, Inpar::S2I::side_slave, islavematrix_, Inpar::S2I::side_slave,
              Inpar::S2I::side_master, imastermatrix_, Inpar::S2I::side_master,
              Inpar::S2I::side_slave, imastermatrix_, Inpar::S2I::side_master,
              Inpar::S2I::side_master,
              islaveresidual_ != nullptr
                  ? Core::Utils::shared_ptr_from_ref(islaveresidual_->as_multi_vector())
                  : nullptr,
              Inpar::S2I::side_slave, imasterresidual_, Inpar::S2I::side_master);
        }
      }

      // finalize auxiliary system matrix for slave side
      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_slave or couplingtype_ == Inpar::S2I::coupling_nts_standard)
        islavematrix_->complete(*interfacemaps_->full_map(), *interfacemaps_->map(1));

      // finalize auxiliary system matrix and residual vector for master side
      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_master or couplingtype_ == Inpar::S2I::coupling_nts_standard)
      {
        imastermatrix_->complete(*interfacemaps_->full_map(), *interfacemaps_->map(2));
        imasterresidual_->complete(Core::LinAlg::CombineMode::add, true);
      }

      // assemble global system of equations depending on matrix type
      switch (matrixtype_)
      {
        case Core::LinAlg::MatrixType::sparse:
        {
          // extract global system matrix from time integrator
          const std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix =
              scatratimint_->system_matrix();
          if (systemmatrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

          // assemble interface contributions into global system of equations
          switch (couplingtype_)
          {
            case Inpar::S2I::coupling_mortar_standard:
            case Inpar::S2I::coupling_nts_standard:
            {
              const std::shared_ptr<const Core::LinAlg::SparseMatrix> islavematrix =
                  not imortarredistribution_
                      ? islavematrix_
                      : Core::LinAlg::matrix_row_transform(*islavematrix_, *islavemap_);
              const std::shared_ptr<const Core::LinAlg::SparseMatrix> imastermatrix =
                  not imortarredistribution_
                      ? imastermatrix_
                      : Core::LinAlg::matrix_row_transform(*imastermatrix_, *imastermap_);
              Core::LinAlg::matrix_add(*islavematrix, false, 1., *systemmatrix, 1.);
              Core::LinAlg::matrix_add(*imastermatrix, false, 1., *systemmatrix, 1.);
              interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
              interfacemaps_->add_vector(
                  imasterresidual_->as_multi_vector(), 2, *scatratimint_->residual());

              break;
            }

            case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
            case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
            {
              if (lmside_ == Inpar::S2I::side_slave)
              {
                // assemble slave-side interface contributions into global residual vector
                Core::LinAlg::Vector<double> islaveresidual(*interfacemaps_->map(1));
                D_->multiply(true, *lm_, islaveresidual);
                interfacemaps_->add_vector(islaveresidual, 1, *scatratimint_->residual(), -1.);

                // assemble master-side interface contributions into global residual vector
                Core::LinAlg::Vector<double> imasterresidual(*interfacemaps_->map(2));
                M_->multiply(true, *lm_, imasterresidual);
                interfacemaps_->add_vector(imasterresidual, 2, *scatratimint_->residual());

                // build constraint residual vector associated with Lagrange multiplier dofs
                Core::LinAlg::Vector<double> ilmresidual(*islaveresidual_);
                ilmresidual.replace_map(*extendedmaps_->map(1));
                lmresidual_->update(1., ilmresidual, 0.);
                E_->multiply(true, *lm_, ilmresidual);
                lmresidual_->update(1., ilmresidual, 1.);
              }
              else
              {
                // assemble slave-side interface contributions into global residual vector
                Core::LinAlg::Vector<double> islaveresidual(*interfacemaps_->map(1));
                M_->multiply(true, *lm_, islaveresidual);
                interfacemaps_->add_vector(islaveresidual, 1, *scatratimint_->residual());

                // assemble master-side interface contributions into global residual vector
                Core::LinAlg::Vector<double> imasterresidual(*interfacemaps_->map(2));
                D_->multiply(true, *lm_, imasterresidual);
                interfacemaps_->add_vector(imasterresidual, 2, *scatratimint_->residual(), -1.);

                // build constraint residual vector associated with Lagrange multiplier dofs
                Core::LinAlg::Vector<double> ilmresidual(*imasterresidual_);
                ilmresidual.replace_map(*extendedmaps_->map(1));
                lmresidual_->update(1., ilmresidual, 0.);
                E_->multiply(true, *lm_, ilmresidual);
                lmresidual_->update(1., ilmresidual, 1.);
              }

              break;
            }

            case Inpar::S2I::coupling_mortar_condensed_petrov:
            {
              if (lmside_ == Inpar::S2I::side_slave)
              {
                Core::LinAlg::matrix_add(*islavematrix_, false, 1., *systemmatrix, 1.);
                Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                             *P_, true, *islavematrix_, false, false, false, true),
                    false, -1., *systemmatrix, 1.);
                interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
                Core::LinAlg::Vector<double> imasterresidual(*interfacemaps_->map(2));
                P_->multiply(true, *islaveresidual_, imasterresidual);
                interfacemaps_->add_vector(imasterresidual, 2, *scatratimint_->residual(), -1.);
              }
              else
              {
                Core::LinAlg::matrix_add(*Core::LinAlg::matrix_multiply(
                                             *P_, true, *imastermatrix_, false, false, false, true),
                    false, -1., *systemmatrix, 1.);
                Core::LinAlg::matrix_add(*imastermatrix_, false, 1., *systemmatrix, 1.);
                Core::LinAlg::Vector<double> islaveresidual(*interfacemaps_->map(1));
                P_->multiply(true, imasterresidual_->as_multi_vector(), islaveresidual);
                interfacemaps_->add_vector(islaveresidual, 1, *scatratimint_->residual(), -1.);
                interfacemaps_->add_vector(
                    imasterresidual_->as_multi_vector(), 2, *scatratimint_->residual());
              }

              break;
            }

            case Inpar::S2I::coupling_mortar_condensed_bubnov:
            {
              // assemble interface contributions into global system of equations
              if (slaveonly_)
              {
                Core::LinAlg::matrix_add(*islavematrix_, false, 1., *systemmatrix, 1.);
                interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
              }

              // during calculation of initial time derivative, condensation must not be performed
              // here, but after assembly of the modified global system of equations
              if (scatratimint_->step() > 0)
                condense_mat_and_rhs(systemmatrix, scatratimint_->residual());

              break;
            }

            default:
            {
              FOUR_C_THROW("Not yet implemented!");
              break;
            }
          }

          break;
        }

        case Core::LinAlg::MatrixType::block_condition:
        {
          // extract global system matrix from time integrator
          std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksystemmatrix =
              scatratimint_->block_system_matrix();
          if (blocksystemmatrix == nullptr) FOUR_C_THROW("System matrix is not a block matrix!");

          // assemble interface contributions into global system of equations
          switch (couplingtype_)
          {
            case Inpar::S2I::coupling_mortar_standard:
            {
              // split interface sparse matrices into block matrices
              const std::shared_ptr<const Core::LinAlg::SparseMatrix> islavematrix =
                  not imortarredistribution_
                      ? islavematrix_
                      : Core::LinAlg::matrix_row_transform(*islavematrix_, *islavemap_);
              std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockslavematrix(
                  Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                      *islavematrix, *scatratimint_->dof_block_maps(), *blockmaps_slave_));
              blockslavematrix->complete();
              const std::shared_ptr<const Core::LinAlg::SparseMatrix> imastermatrix =
                  not imortarredistribution_
                      ? imastermatrix_
                      : Core::LinAlg::matrix_row_transform(*imastermatrix_, *imastermap_);
              std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockmastermatrix(
                  Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                      *imastermatrix, *scatratimint_->dof_block_maps(), *blockmaps_master_));
              blockmastermatrix->complete();

              // assemble interface block matrices into global block system matrix
              blocksystemmatrix->add(*blockslavematrix, false, 1., 1.);
              blocksystemmatrix->add(*blockmastermatrix, false, 1., 1.);

              // assemble interface residual vectors into global residual vector
              interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
              interfacemaps_->add_vector(
                  Core::LinAlg::MultiVector<double>(imasterresidual_->as_multi_vector()), 2,
                  *scatratimint_->residual());

              break;
            }

            default:
            {
              FOUR_C_THROW("Not yet implemented!");
              break;
            }
          }

          break;
        }

        default:
        {
          FOUR_C_THROW("Not yet implemented!");
          break;
        }
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Not yet implemented!");
    }
  }
  // extract boundary conditions for scatra-scatra interface layer growth
  std::vector<const Core::Conditions::Condition*> s2icoupling_growth_conditions;
  scatratimint_->discretization()->get_condition(
      "S2IKineticsGrowth", s2icoupling_growth_conditions);

  // evaluate scatra-scatra interface layer growth
  if (s2icoupling_growth_conditions.size())
  {
    switch (couplingtype_)
    {
      case Inpar::S2I::coupling_matching_nodes:
      {
        // create parameter list for elements
        Teuchos::ParameterList conditionparams;

        // action for elements
        Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
            "action", ScaTra::BoundaryAction::calc_s2icoupling, conditionparams);

        // set global state vectors according to time-integration scheme
        scatratimint_->add_time_integration_specific_vectors();

        // evaluate scatra-scatra interface coupling at time t_{n+1} or t_{n+alpha_F}
        islavematrix_->zero();
        imastermatrix_->zero();
        islaveresidual_->put_scalar(0.);

        // collect condition specific data and store to scatra boundary parameter class
        set_condition_specific_scatra_parameters(*s2icoupling_growth_conditions[0]);
        // evaluate the condition
        scatratimint_->discretization()->evaluate_condition(conditionparams, islavematrix_,
            imastermatrix_, islaveresidual_, nullptr, nullptr, "S2IKineticsGrowth");

        // finalize interface matrices
        islavematrix_->complete();
        imastermatrix_->complete();

        // assemble interface matrices into global system matrix depending on matrix type
        switch (matrixtype_)
        {
          case Core::LinAlg::MatrixType::sparse:
          {
            // check matrix
            const std::shared_ptr<Core::LinAlg::SparseMatrix>& systemmatrix =
                scatratimint_->system_matrix();
            if (systemmatrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

            // We assume that the scatra-scatra interface layer growth is caused by master-side
            // fluxes to the interface, whereas there is no mass exchange between the interface
            // layer and the slave side of the interface. Hence, we only need to linearize the
            // master-side fluxes w.r.t. the slave-side and master-side degrees of freedom.

            // derive linearizations of master fluxes w.r.t. slave dofs and assemble into global
            // system matrix
            Coupling::Adapter::MatrixRowTransform()(*islavematrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true);

            // derive linearizations of master fluxes w.r.t. master dofs and assemble into global
            // system matrix
            Coupling::Adapter::MatrixRowColTransform()(*imastermatrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_),
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true, true);

            break;
          }

          case Core::LinAlg::MatrixType::block_condition:
          case Core::LinAlg::MatrixType::block_condition_dof:
          {
            // check matrix
            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksystemmatrix =
                scatratimint_->block_system_matrix();
            if (blocksystemmatrix == nullptr) FOUR_C_THROW("System matrix is not a block matrix!");

            // We assume that the scatra-scatra interface layer growth is caused by master-side
            // fluxes to the interface, whereas there is no mass exchange between the interface
            // layer and the slave side of the interface. Hence, we only need to linearize the
            // master-side fluxes w.r.t. the slave-side and master-side degrees of freedom.

            // derive linearizations of master fluxes w.r.t. slave dofs
            std::shared_ptr<Core::LinAlg::SparseMatrix> kms(
                std::make_shared<Core::LinAlg::SparseMatrix>(*icoup_->master_dof_map(), 81, false));
            Coupling::Adapter::MatrixRowTransform()(
                *islavematrix_, -1., Coupling::Adapter::CouplingSlaveConverter(*icoup_), *kms);
            kms->complete(*icoup_->slave_dof_map(), *icoup_->master_dof_map());
            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkms(
                Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                    *kms, *blockmaps_slave_, *blockmaps_master_));
            blockkms->complete();

            // derive linearizations of master fluxes w.r.t. master dofs
            std::shared_ptr<Core::LinAlg::SparseMatrix> kmm(
                std::make_shared<Core::LinAlg::SparseMatrix>(*icoup_->master_dof_map(), 81, false));
            Coupling::Adapter::MatrixRowColTransform()(*imastermatrix_, -1.,
                Coupling::Adapter::CouplingSlaveConverter(*icoup_),
                Coupling::Adapter::CouplingSlaveConverter(*icoup_), *kmm);
            kmm->complete();
            std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkmm(
                Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                    *kmm, *blockmaps_master_, *blockmaps_master_));
            blockkmm->complete();

            // assemble interface block matrices into global block system matrix
            blocksystemmatrix->add(*blockkms, false, 1., 1.);
            blocksystemmatrix->add(*blockkmm, false, 1., 1.);

            break;
          }

          default:
          {
            FOUR_C_THROW(
                "Type of global system matrix for scatra-scatra interface coupling involving "
                "interface layer growth not recognized!");
            break;
          }
        }

        // As before, we only need to consider residual contributions from the master-side fluxes.

        // transform master residuals and assemble into global residual vector
        interfacemaps_->add_vector(
            *icoup_->slave_to_master(*islaveresidual_), 2, *scatratimint_->residual(), -1.);

        // compute additional linearizations and residuals in case of monolithic evaluation approach
        if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
        {
          // extract map associated with scalar transport degrees of freedom
          const Core::LinAlg::Map& dofrowmap_scatra =
              *scatratimint_->discretization()->dof_row_map();

          // extract map associated with scatra-scatra interface layer thicknesses
          const Core::LinAlg::Map& dofrowmap_growth =
              *scatratimint_->discretization()->dof_row_map(2);

          // extract ID of boundary condition for scatra-scatra interface layer growth
          // the corresponding boundary condition for scatra-scatra interface coupling is expected
          // to have the same ID
          std::vector<const Core::Conditions::Condition*> growth_conditions;
          scatratimint_->discretization()->get_condition("S2IKineticsGrowth", growth_conditions);
          const int condid = growth_conditions.front()->parameters().get<int>("ConditionID");

          // set global state vectors according to time-integration scheme
          scatratimint_->add_time_integration_specific_vectors();

          // compute additional linearizations and residuals depending on type of scalar transport
          // system matrix
          switch (matrixtype_)
          {
            case Core::LinAlg::MatrixType::sparse:
            {
              // assemble off-diagonal scatra-growth block of global system matrix, containing
              // derivatives of discrete scatra residuals w.r.t. discrete scatra-scatra interface
              // layer thicknesses
              {
                // check matrix
                const std::shared_ptr<Core::LinAlg::SparseMatrix> scatragrowthblock =
                    std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(scatragrowthblock_);
                if (scatragrowthblock == nullptr) FOUR_C_THROW("Matrix is not a sparse matrix!");

                // initialize matrix block
                scatragrowthblock->zero();

                // initialize auxiliary matrix block for linearizations of slave fluxes w.r.t.
                // scatra-scatra interface layer thicknesses
                std::shared_ptr<Core::LinAlg::SparseMatrix> islavematrix =
                    std::make_shared<Core::LinAlg::SparseMatrix>(*(icoup_)->slave_dof_map(), 81);

                // initialize assembly strategy for auxiliary matrix block
                Core::FE::AssembleStrategy strategy(
                    0, 2, islavematrix, nullptr, nullptr, nullptr, nullptr);

                // create parameter list for elements
                Teuchos::ParameterList condparams;

                // set action for elements
                Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
                    "action", ScaTra::BoundaryAction::calc_s2icoupling_scatragrowth, condparams);

                // evaluate off-diagonal linearizations arising from scatra-scatra interface
                // coupling
                for (auto [kinetics_slave_cond_id, kinetics_slave_cond] :
                    kinetics_conditions_meshtying_slaveside_)
                {
                  if (kinetics_slave_cond_id == condid)
                  {
                    // collect condition specific data and store to scatra boundary parameter class
                    set_condition_specific_scatra_parameters(*kinetics_slave_cond);

                    scatratimint_->discretization()->evaluate_condition(
                        condparams, strategy, "S2IKinetics", condid);
                  }
                }

                // finalize auxiliary matrix block
                islavematrix->complete(dofrowmap_growth, dofrowmap_scatra);

                // assemble linearizations of slave fluxes associated with scatra-scatra interface
                // coupling w.r.t. scatra-scatra interface layer thicknesses into global matrix
                // block
                Core::LinAlg::matrix_add(*islavematrix, false, 1., *scatragrowthblock, 0.);

                // derive linearizations of master fluxes associated with scatra-scatra interface
                // coupling w.r.t. scatra-scatra interface layer thicknesses and assemble into
                // global matrix block
                Coupling::Adapter::MatrixRowTransform()(*islavematrix, -1.,
                    Coupling::Adapter::CouplingSlaveConverter(*icoup_), *scatragrowthblock, false);

                // zero out auxiliary matrix block for subsequent evaluation
                islavematrix->zero();

                // evaluate off-diagonal linearizations arising from scatra-scatra interface layer
                // growth
                std::vector<const Core::Conditions::Condition*> conds;
                scatratimint_->discretization()->get_condition("S2IKineticsGrowth", conds);
                FOUR_C_ASSERT_ALWAYS(conds.size() == 1,
                    "There should be only one condition for interface layer growth!");
                const Core::Conditions::Condition* s2i_coupling_growth_cond = conds[0];

                // collect condition specific data and store to scatra boundary parameter class
                set_condition_specific_scatra_parameters(*s2i_coupling_growth_cond);

                scatratimint_->discretization()->evaluate_condition(
                    condparams, strategy, "S2IKineticsGrowth", condid);

                // finalize auxiliary matrix block
                islavematrix->complete(dofrowmap_growth, dofrowmap_scatra);

                // derive linearizations of master fluxes associated with scatra-scatra interface
                // layer growth w.r.t. scatra-scatra interface layer thicknesses and assemble into
                // global matrix block
                Coupling::Adapter::MatrixRowTransform()(*islavematrix, -1.,
                    Coupling::Adapter::CouplingSlaveConverter(*icoup_), *scatragrowthblock, true);

                // finalize global matrix block
                scatragrowthblock->complete(dofrowmap_growth, dofrowmap_scatra);

                // apply Dirichlet boundary conditions to global matrix block
                scatragrowthblock->apply_dirichlet(
                    *scatratimint_->dirich_maps()->cond_map(), false);
              }

              // assemble off-diagonal growth-scatra block of global system matrix, containing
              // derivatives of discrete scatra-scatra interface layer growth residuals w.r.t.
              // discrete scatra degrees of freedom
              {
                // check matrix
                const std::shared_ptr<Core::LinAlg::SparseMatrix> growthscatrablock =
                    std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(growthscatrablock_);
                if (growthscatrablock == nullptr) FOUR_C_THROW("Matrix is not a sparse matrix!");

                // initialize matrix block
                growthscatrablock->zero();

                // initialize auxiliary matrix blocks for linearizations of scatra-scatra interface
                // layer growth residuals w.r.t. slave-side and master-side scalar transport degrees
                // of freedom
                std::shared_ptr<Core::LinAlg::SparseMatrix> islavematrix =
                    std::make_shared<Core::LinAlg::SparseMatrix>(dofrowmap_growth, 81);
                std::shared_ptr<Core::LinAlg::SparseMatrix> imastermatrix =
                    std::make_shared<Core::LinAlg::SparseMatrix>(dofrowmap_growth, 81);

                // initialize assembly strategy for auxiliary matrix block
                Core::FE::AssembleStrategy strategy(
                    2, 0, islavematrix, imastermatrix, nullptr, nullptr, nullptr);

                // create parameter list for elements
                Teuchos::ParameterList condparams;

                // set action for elements
                Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
                    "action", ScaTra::BoundaryAction::calc_s2icoupling_growthscatra, condparams);

                // evaluate off-diagonal linearizations
                scatratimint_->discretization()->evaluate_condition(
                    condparams, strategy, "S2IKineticsGrowth", condid);

                // finalize auxiliary matrix blocks
                islavematrix->complete(dofrowmap_scatra, dofrowmap_growth);
                imastermatrix->complete(dofrowmap_scatra, dofrowmap_growth);

                // assemble linearizations of scatra-scatra interface layer growth residuals w.r.t.
                // slave-side scalar transport degrees of freedom into global matrix block
                Core::LinAlg::matrix_add(*islavematrix, false, 1., *growthscatrablock, 0.);

                // derive linearizations of scatra-scatra interface layer growth residuals w.r.t.
                // master-side scalar transport degrees of freedom and assemble into global matrix
                // block
                Coupling::Adapter::MatrixColTransform()(imastermatrix->row_map(),
                    imastermatrix->col_map(), *imastermatrix, 1.,
                    Coupling::Adapter::CouplingSlaveConverter(*icoup_), *growthscatrablock, true,
                    true);

                // finalize global matrix block
                growthscatrablock->complete(dofrowmap_scatra, dofrowmap_growth);
              }

              break;
            }

            case Core::LinAlg::MatrixType::block_condition:
            case Core::LinAlg::MatrixType::block_condition_dof:
            {
              // assemble off-diagonal scatra-growth block of global system matrix, containing
              // derivatives of discrete scatra residuals w.r.t. discrete scatra-scatra interface
              // layer thicknesses
              {
                // initialize auxiliary matrix block for linearizations of slave fluxes w.r.t.
                // scatra-scatra interface layer thicknesses
                const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockslavematrix =
                    std::make_shared<
                        Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                        *blockmapgrowth_, *blockmaps_slave_, 81, false, true);

                // initialize assembly strategy for auxiliary matrix block
                Core::FE::AssembleStrategy strategy(
                    0, 2, blockslavematrix, nullptr, nullptr, nullptr, nullptr);

                // create parameter list for elements
                Teuchos::ParameterList condparams;

                // set action for elements
                Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
                    "action", ScaTra::BoundaryAction::calc_s2icoupling_scatragrowth, condparams);

                // evaluate off-diagonal linearizations arising from scatra-scatra interface
                // coupling
                scatratimint_->discretization()->evaluate_condition(
                    condparams, strategy, "S2IKinetics", condid);

                // finalize auxiliary matrix block
                blockslavematrix->complete();

                // assemble linearizations of slave fluxes associated with scatra-scatra interface
                // coupling w.r.t. scatra-scatra interface layer thicknesses into global matrix
                // block
                scatragrowthblock_->add(*blockslavematrix, false, 1., 0.);

                // initialize auxiliary system matrix for linearizations of master fluxes associated
                // with scatra-scatra interface coupling w.r.t. scatra-scatra interface layer
                // thicknesses
                Core::LinAlg::SparseMatrix mastermatrix(*icoup_->master_dof_map(), 27, false, true);

                // derive linearizations of master fluxes associated with scatra-scatra interface
                // coupling w.r.t. scatra-scatra interface layer thicknesses
                for (int iblock = 0; iblock < blockmaps_slave_->num_maps(); ++iblock)
                  Coupling::Adapter::MatrixRowTransform()(blockslavematrix->matrix(iblock, 0), -1.,
                      Coupling::Adapter::CouplingSlaveConverter(*icoup_), mastermatrix, true);

                // zero out auxiliary matrices for subsequent evaluation
                blockslavematrix->zero();
                mastermatrix.zero();

                // evaluate off-diagonal linearizations arising from scatra-scatra interface layer
                // growth
                scatratimint_->discretization()->evaluate_condition(
                    condparams, strategy, "S2IKineticsGrowth", condid);

                // derive linearizations of master fluxes associated with scatra-scatra interface
                // layer growth w.r.t. scatra-scatra interface layer thicknesses
                for (int iblock = 0; iblock < blockmaps_slave_->num_maps(); ++iblock)
                  Coupling::Adapter::MatrixRowTransform()(blockslavematrix->matrix(iblock, 0), -1.,
                      Coupling::Adapter::CouplingSlaveConverter(*icoup_), mastermatrix, true);

                // finalize auxiliary system matrix
                mastermatrix.complete(dofrowmap_growth, *icoup_->master_dof_map());

                // split auxiliary system matrix and assemble into global matrix block
                const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockmastermatrix =
                    Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                        mastermatrix, *blockmapgrowth_, *scatratimint_->dof_block_maps());
                blockmastermatrix->complete();
                scatragrowthblock_->add(*blockmastermatrix, false, 1., 1.);

                // finalize global matrix block
                scatragrowthblock_->complete();

                // apply Dirichlet boundary conditions to global matrix block
                scatragrowthblock_->apply_dirichlet(
                    *scatratimint_->dirich_maps()->cond_map(), false);
              }

              // assemble off-diagonal growth-scatra block of global system matrix, containing
              // derivatives of discrete scatra-scatra interface layer growth residuals w.r.t.
              // discrete scatra degrees of freedom
              {
                // initialize auxiliary matrix blocks for linearizations of scatra-scatra interface
                // layer growth residuals w.r.t. slave-side and master-side scalar transport degrees
                // of freedom
                const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockslavematrix =
                    std::make_shared<
                        Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                        *blockmaps_slave_, *blockmapgrowth_, 81, false, true);
                const std::shared_ptr<Core::LinAlg::SparseMatrix> imastermatrix =
                    std::make_shared<Core::LinAlg::SparseMatrix>(dofrowmap_growth, 81);

                // initialize assembly strategy for auxiliary matrix blocks
                Core::FE::AssembleStrategy strategy(
                    2, 0, blockslavematrix, imastermatrix, nullptr, nullptr, nullptr);

                // create parameter list for elements
                Teuchos::ParameterList condparams;

                // set action for elements
                Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
                    "action", ScaTra::BoundaryAction::calc_s2icoupling_growthscatra, condparams);

                // evaluate off-diagonal linearizations
                scatratimint_->discretization()->evaluate_condition(
                    condparams, strategy, "S2IKineticsGrowth", condid);

                // finalize auxiliary matrix blocks
                blockslavematrix->complete();
                imastermatrix->complete(dofrowmap_scatra, dofrowmap_growth);

                // assemble linearizations of scatra-scatra interface layer growth residuals w.r.t.
                // slave-side scalar transport degrees of freedom into global matrix block
                growthscatrablock_->add(*blockslavematrix, false, 1., 0.);

                // initialize temporary matrix
                Core::LinAlg::SparseMatrix kgm(dofrowmap_growth, 27, false, true);

                // derive linearizations of scatra-scatra interface layer growth residuals w.r.t.
                // master-side scalar transport degrees of freedom
                Coupling::Adapter::MatrixColTransform()(imastermatrix->row_map(),
                    imastermatrix->col_map(), *imastermatrix, 1.,
                    Coupling::Adapter::CouplingSlaveConverter(*icoup_), kgm);

                // finalize temporary matrix
                kgm.complete(*icoup_->master_dof_map(), dofrowmap_growth);

                // split temporary matrix and assemble into global matrix block
                const std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blockkgm(
                    Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
                        kgm, *scatratimint_->dof_block_maps(), *blockmapgrowth_));
                blockkgm->complete();
                growthscatrablock_->add(*blockkgm, false, 1., 1.);

                // finalize global matrix block
                growthscatrablock_->complete();
              }

              break;
            }

            default:
            {
              FOUR_C_THROW(
                  "Type of global system matrix for scatra-scatra interface coupling involving "
                  "interface layer growth not recognized!");
              break;
            }
          }  // type of scalar transport system matrix

          // assemble residual vector associated with scatra-scatra interface layer thicknesses and
          // main-diagonal growth-growth block of global system matrix, containing derivatives of
          // discrete scatra-scatra interface layer growth residuals w.r.t. discrete scatra-scatra
          // interface layer thicknesses
          {
            // initialize matrix block and corresponding residual vector
            growthgrowthblock_->zero();
            growthresidual_->put_scalar(0.);

            // initialize assembly strategy for main-diagonal growth-growth block and
            Core::FE::AssembleStrategy strategy(
                2, 2, growthgrowthblock_, nullptr, growthresidual_, nullptr, nullptr);

            // create parameter list for elements
            Teuchos::ParameterList condparams;

            // set action for elements
            Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
                "action", ScaTra::BoundaryAction::calc_s2icoupling_growthgrowth, condparams);

            // set history vector associated with discrete scatra-scatra interface layer thicknesses
            scatratimint_->discretization()->set_state(2, "growthhist", *growthhist_);

            // evaluate main-diagonal linearizations and corresponding residuals
            scatratimint_->discretization()->evaluate_condition(
                condparams, strategy, "S2IKineticsGrowth", condid);

            // finalize global matrix block
            growthgrowthblock_->complete();
          }
        }  // monolithic evaluation of scatra-scatra interface layer growth

        break;
      }

      default:
      {
        FOUR_C_THROW(
            "Evaluation of scatra-scatra interface layer growth only implemented for conforming "
            "interface discretizations!");
        break;
      }
    }
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_and_assemble_capacitive_contributions()
{
  // create parameter list for elements
  Teuchos::ParameterList capcondparas;

  // action for elements
  Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
      "action", ScaTra::BoundaryAction::calc_s2icoupling_capacitance, capcondparas);

  // set global state vectors according to time-integration scheme
  scatratimint_->add_time_integration_specific_vectors();

  // zero out matrices and vectors
  islavematrix_->zero();
  imasterslavematrix_->zero();
  islaveresidual_->put_scalar(0.0);
  auto imasterresidual_on_slave_side =
      std::make_shared<Core::LinAlg::Vector<double>>(*interfacemaps_->map(1));
  imasterresidual_on_slave_side->put_scalar(0.0);

  // evaluate scatra-scatra interface coupling
  for (auto kinetics_slave_cond_cap : kinetics_conditions_meshtying_slaveside_)
  {
    if (kinetics_slave_cond_cap.second->parameters().get<Inpar::S2I::KineticModels>(
            "KINETIC_MODEL") ==
        static_cast<int>(Inpar::S2I::kinetics_butlervolmerreducedcapacitance))
    {
      // collect condition specific data and store to scatra boundary parameter class
      set_condition_specific_scatra_parameters(*kinetics_slave_cond_cap.second);

      scatratimint_->discretization()->evaluate_condition(capcondparas, islavematrix_,
          imasterslavematrix_, islaveresidual_, imasterresidual_on_slave_side, nullptr,
          "S2IKinetics", kinetics_slave_cond_cap.second->parameters().get<int>("ConditionID"));
    }
  }

  // finalize interface matrices
  islavematrix_->complete();
  imasterslavematrix_->complete();

  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      auto systemmatrix = scatratimint_->system_matrix();
      FOUR_C_ASSERT(systemmatrix != nullptr, "System matrix is not a sparse matrix!");

      // assemble additional components of linearizations of slave fluxes due to capacitance
      // w.r.t. slave dofs into the global system matrix
      Core::LinAlg::matrix_add(*islavematrix_, false, 1.0, *systemmatrix, 1.0);

      // assemble additional components of linearizations of slave fluxes due to capacitance
      // w.r.t. master dofs into the global system matrix
      Coupling::Adapter::MatrixColTransform()(islavematrix_->row_map(), islavematrix_->col_map(),
          *islavematrix_, -1.0, Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix,
          true, true);

      // assemble additional components of linearizations of master fluxes due to capacitance
      // w.r.t. slave dofs into the global system matrix
      Coupling::Adapter::MatrixRowTransform()(*imasterslavematrix_, 1.0,
          Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true);

      // assemble additional components of linearizations of master fluxes due to capacitance
      // w.r.t. master dofs into the global system matrix
      Coupling::Adapter::MatrixRowColTransform()(*imasterslavematrix_, -1.0,
          Coupling::Adapter::CouplingSlaveConverter(*icoup_),
          Coupling::Adapter::CouplingSlaveConverter(*icoup_), *systemmatrix, true, true);
      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      // check matrix
      auto blocksystemmatrix = scatratimint_->block_system_matrix();
      FOUR_C_ASSERT(blocksystemmatrix != nullptr, "System matrix is not a block matrix!");

      // prepare linearizations of slave fluxes due to capacitance w.r.t. slave dofs
      auto blockkss = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *islavematrix_, *blockmaps_slave_, *blockmaps_slave_);
      blockkss->complete();

      // prepare linearizations of slave fluxes due to capacitance w.r.t. master dofs
      Core::LinAlg::SparseMatrix ksm(*icoup_->slave_dof_map(), 81, false);
      Coupling::Adapter::MatrixColTransform()(islavematrix_->row_map(), islavematrix_->col_map(),
          *islavematrix_, -1.0, Coupling::Adapter::CouplingSlaveConverter(*icoup_), ksm);
      ksm.complete(*icoup_->master_dof_map(), *icoup_->slave_dof_map());
      auto blockksm = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          ksm, *blockmaps_master_, *blockmaps_slave_);
      blockksm->complete();

      // prepare linearizations of master fluxes due to capacitance w.r.t. slave dofs
      Core::LinAlg::SparseMatrix kms(*icoup_->master_dof_map(), 81, false);
      Coupling::Adapter::MatrixRowTransform()(
          *imasterslavematrix_, 1.0, Coupling::Adapter::CouplingSlaveConverter(*icoup_), kms);
      kms.complete(*icoup_->slave_dof_map(), *icoup_->master_dof_map());
      auto blockkms = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          kms, *blockmaps_slave_, *blockmaps_master_);
      blockkms->complete();

      // derive linearizations of master fluxes w.r.t. master dofs
      Core::LinAlg::SparseMatrix kmm(*icoup_->master_dof_map(), 81, false);
      Coupling::Adapter::MatrixRowColTransform()(*imasterslavematrix_, -1.0,
          Coupling::Adapter::CouplingSlaveConverter(*icoup_),
          Coupling::Adapter::CouplingSlaveConverter(*icoup_), kmm);
      kmm.complete();
      auto blockkmm = Core::LinAlg::split_matrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          kmm, *blockmaps_master_, *blockmaps_master_);
      blockkmm->complete();

      // assemble interface block matrices into global block system matrix
      blocksystemmatrix->add(*blockkss, false, 1.0, 1.0);
      blocksystemmatrix->add(*blockksm, false, 1.0, 1.0);
      blocksystemmatrix->add(*blockkms, false, 1.0, 1.0);
      blocksystemmatrix->add(*blockkmm, false, 1.0, 1.0);

      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Type of global system matrix for scatra-scatra interface coupling not recognized!");
      break;
    }
  }

  // assemble slave residuals into global residual vector
  interfacemaps_->add_vector(*islaveresidual_, 1, *scatratimint_->residual());
  // transform master residuals and assemble into global residual vector
  interfacemaps_->add_vector(
      *icoup_->slave_to_master(*imasterresidual_on_slave_side), 2, *scatratimint_->residual(), 1.0);
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_mortar_cell(const Core::FE::Discretization& idiscret,
    Mortar::IntCell& cell, const Inpar::ScaTra::ImplType& impltype, Mortar::Element& slaveelement,
    Mortar::Element& masterelement, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& cellmatrix1, Core::LinAlg::SerialDenseMatrix& cellmatrix2,
    Core::LinAlg::SerialDenseMatrix& cellmatrix3, Core::LinAlg::SerialDenseMatrix& cellmatrix4,
    Core::LinAlg::SerialDenseVector& cellvector1,
    Core::LinAlg::SerialDenseVector& cellvector2) const
{
  // evaluate single mortar integration cell
  ScaTra::MortarCellFactory::mortar_cell_calc(
      impltype, slaveelement, masterelement, couplingtype_, lmside_, idiscret.name())
      ->evaluate(idiscret, cell, slaveelement, masterelement, la_slave, la_master, params,
          cellmatrix1, cellmatrix2, cellmatrix3, cellmatrix4, cellvector1, cellvector2);
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_slave_node(const Core::FE::Discretization& idiscret,
    const Mortar::Node& slavenode, const double& lumpedarea,
    const Inpar::ScaTra::ImplType& impltype, Mortar::Element& slaveelement,
    Mortar::Element& masterelement, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& ntsmatrix1, Core::LinAlg::SerialDenseMatrix& ntsmatrix2,
    Core::LinAlg::SerialDenseMatrix& ntsmatrix3, Core::LinAlg::SerialDenseMatrix& ntsmatrix4,
    Core::LinAlg::SerialDenseVector& ntsvector1, Core::LinAlg::SerialDenseVector& ntsvector2) const
{
  // evaluate single slave-side node
  ScaTra::MortarCellFactory::mortar_cell_calc(
      impltype, slaveelement, masterelement, couplingtype_, lmside_, idiscret.name())
      ->evaluate_nts(idiscret, slavenode, lumpedarea, slaveelement, masterelement, la_slave,
          la_master, params, ntsmatrix1, ntsmatrix2, ntsmatrix3, ntsmatrix4, ntsvector1,
          ntsvector2);
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_mortar_element(const Core::FE::Discretization& idiscret,
    Mortar::Element& element, const Inpar::ScaTra::ImplType& impltype,
    Core::Elements::LocationArray& la, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseMatrix& elematrix3, Core::LinAlg::SerialDenseMatrix& elematrix4,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2) const
{
  // evaluate single mortar element
  ScaTra::MortarCellFactory::mortar_cell_calc(
      impltype, element, element, couplingtype_, lmside_, idiscret.name())
      ->evaluate_mortar_element(idiscret, element, la, params, elematrix1, elematrix2, elematrix3,
          elematrix4, elevector1, elevector2);
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_mortar_cells(const Core::FE::Discretization& idiscret,
    const Teuchos::ParameterList& params,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix1,
    const Inpar::S2I::InterfaceSides matrix1_side_rows,
    const Inpar::S2I::InterfaceSides matrix1_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix2,
    const Inpar::S2I::InterfaceSides matrix2_side_rows,
    const Inpar::S2I::InterfaceSides matrix2_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix3,
    const Inpar::S2I::InterfaceSides matrix3_side_rows,
    const Inpar::S2I::InterfaceSides matrix3_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix4,
    const Inpar::S2I::InterfaceSides matrix4_side_rows,
    const Inpar::S2I::InterfaceSides matrix4_side_cols,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& systemvector1,
    const Inpar::S2I::InterfaceSides vector1_side,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& systemvector2,
    const Inpar::S2I::InterfaceSides vector2_side) const
{
  // instantiate assembly strategy for mortar integration cells
  ScaTra::MortarCellAssemblyStrategy strategy(systemmatrix1, matrix1_side_rows, matrix1_side_cols,
      systemmatrix2, matrix2_side_rows, matrix2_side_cols, systemmatrix3, matrix3_side_rows,
      matrix3_side_cols, systemmatrix4, matrix4_side_rows, matrix4_side_cols, systemvector1,
      vector1_side, systemvector2, vector2_side);

  // evaluate mortar integration cells
  evaluate_mortar_cells(idiscret, params, strategy);
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_mortar_cells(const Core::FE::Discretization& idiscret,
    const Teuchos::ParameterList& params, ScaTra::MortarCellAssemblyStrategy& strategy) const
{
  // extract scatra-scatra interface coupling condition from parameter list
  const Core::Conditions::Condition* const condition =
      params.get<const Core::Conditions::Condition*>("condition");
  if (condition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // extract mortar integration cells associated with current condition
  const std::vector<std::pair<std::shared_ptr<Mortar::IntCell>, Inpar::ScaTra::ImplType>>& cells =
      imortarcells_.at(condition->parameters().get<int>("ConditionID"));

  // loop over all mortar integration cells
  for (const auto& icell : cells)
  {
    // extract current mortar integration cell
    const std::shared_ptr<Mortar::IntCell>& cell = icell.first;
    if (cell == nullptr) FOUR_C_THROW("Invalid mortar integration cell!");

    // extract slave-side element associated with current cell
    auto* slaveelement = dynamic_cast<Mortar::Element*>(idiscret.g_element(cell->get_slave_id()));
    if (!slaveelement)
      FOUR_C_THROW("Couldn't extract slave element from mortar interface discretization!");

    // extract master-side element associated with current cell
    auto* masterelement = dynamic_cast<Mortar::Element*>(idiscret.g_element(cell->get_master_id()));
    if (!masterelement)
      FOUR_C_THROW("Couldn't extract master element from mortar interface discretization!");

    // safety check
    if (!slaveelement->is_slave() or masterelement->is_slave())
      FOUR_C_THROW("Something is wrong with the slave-master element pairing!");

    // construct slave-side and master-side location arrays
    Core::Elements::LocationArray la_slave(idiscret.num_dof_sets());
    slaveelement->location_vector(idiscret, la_slave);
    Core::Elements::LocationArray la_master(idiscret.num_dof_sets());
    masterelement->location_vector(idiscret, la_master);

    // initialize cell matrices and vectors
    strategy.init_cell_matrices_and_vectors(la_slave, la_master);

    // evaluate current cell
    evaluate_mortar_cell(idiscret, *cell, icell.second, *slaveelement, *masterelement, la_slave,
        la_master, params, strategy.cell_matrix1(), strategy.cell_matrix2(),
        strategy.cell_matrix3(), strategy.cell_matrix4(), strategy.cell_vector1(),
        strategy.cell_vector2());

    // assemble cell matrices and vectors into system matrices and vectors
    strategy.assemble_cell_matrices_and_vectors(la_slave, la_master, la_slave[0].lmowner_[0]);
  }
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_nts(
    const Core::LinAlg::Vector<int>& islavenodestomasterelements,
    const Core::LinAlg::Vector<double>& islavenodeslumpedareas,
    const Core::LinAlg::Vector<int>& islavenodesimpltypes, const Core::FE::Discretization& idiscret,
    const Teuchos::ParameterList& params,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix1,
    const Inpar::S2I::InterfaceSides matrix1_side_rows,
    const Inpar::S2I::InterfaceSides matrix1_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix2,
    const Inpar::S2I::InterfaceSides matrix2_side_rows,
    const Inpar::S2I::InterfaceSides matrix2_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix3,
    const Inpar::S2I::InterfaceSides matrix3_side_rows,
    const Inpar::S2I::InterfaceSides matrix3_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix4,
    const Inpar::S2I::InterfaceSides matrix4_side_rows,
    const Inpar::S2I::InterfaceSides matrix4_side_cols,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& systemvector1,
    const Inpar::S2I::InterfaceSides vector1_side,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& systemvector2,
    const Inpar::S2I::InterfaceSides vector2_side) const
{
  // instantiate assembly strategy for node-to-segment coupling
  ScaTra::MortarCellAssemblyStrategy strategy(systemmatrix1, matrix1_side_rows, matrix1_side_cols,
      systemmatrix2, matrix2_side_rows, matrix2_side_cols, systemmatrix3, matrix3_side_rows,
      matrix3_side_cols, systemmatrix4, matrix4_side_rows, matrix4_side_cols, systemvector1,
      vector1_side, systemvector2, vector2_side);

  // extract slave-side noderowmap
  const Core::LinAlg::Map& noderowmap_slave = islavenodestomasterelements.get_map();

  // loop over all slave-side nodes
  for (int inode = 0; inode < noderowmap_slave.num_my_elements(); ++inode)
  {
    // extract slave-side node
    auto* const slavenode =
        dynamic_cast<Mortar::Node* const>(idiscret.g_node(noderowmap_slave.gid(inode)));
    if (slavenode == nullptr) FOUR_C_THROW("Couldn't extract slave-side node from discretization!");

    // extract first slave-side element associated with current slave-side node
    auto* const slaveelement =
        dynamic_cast<Mortar::Element* const>(slavenode->adjacent_elements()[0].user_element());
    if (!slaveelement) FOUR_C_THROW("Invalid slave-side mortar element!");

    // extract master-side element associated with current slave-side node
    auto* const masterelement = dynamic_cast<Mortar::Element* const>(
        idiscret.g_element(islavenodestomasterelements.get_local_values()[inode]));
    if (!masterelement) FOUR_C_THROW("Invalid master-side mortar element!");

    // safety check
    if (!slaveelement->is_slave() or masterelement->is_slave())
      FOUR_C_THROW("Something is wrong with the slave-master element pairing!");

    // construct slave-side and master-side location arrays
    Core::Elements::LocationArray la_slave(idiscret.num_dof_sets());
    slaveelement->location_vector(idiscret, la_slave);
    Core::Elements::LocationArray la_master(idiscret.num_dof_sets());
    masterelement->location_vector(idiscret, la_master);

    // initialize cell matrices and vectors
    strategy.init_cell_matrices_and_vectors(la_slave, la_master);

    // evaluate current slave-side node
    evaluate_slave_node(idiscret, *slavenode, islavenodeslumpedareas.local_values_as_span()[inode],
        (Inpar::ScaTra::ImplType)islavenodesimpltypes.get_local_values()[inode], *slaveelement,
        *masterelement, la_slave, la_master, params, strategy.cell_matrix1(),
        strategy.cell_matrix2(), strategy.cell_matrix3(), strategy.cell_matrix4(),
        strategy.cell_vector1(), strategy.cell_vector2());

    // assemble cell matrices and vectors into system matrices and vectors
    strategy.assemble_cell_matrices_and_vectors(la_slave, la_master, slavenode->owner());
  }
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::evaluate_mortar_elements(const Core::LinAlg::Map& ielecolmap,
    const Core::LinAlg::Vector<int>& ieleimpltypes, const Core::FE::Discretization& idiscret,
    const Teuchos::ParameterList& params,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix1,
    const Inpar::S2I::InterfaceSides matrix1_side_rows,
    const Inpar::S2I::InterfaceSides matrix1_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix2,
    const Inpar::S2I::InterfaceSides matrix2_side_rows,
    const Inpar::S2I::InterfaceSides matrix2_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix3,
    const Inpar::S2I::InterfaceSides matrix3_side_rows,
    const Inpar::S2I::InterfaceSides matrix3_side_cols,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix4,
    const Inpar::S2I::InterfaceSides matrix4_side_rows,
    const Inpar::S2I::InterfaceSides matrix4_side_cols,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& systemvector1,
    const Inpar::S2I::InterfaceSides vector1_side,
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& systemvector2,
    const Inpar::S2I::InterfaceSides vector2_side) const
{
  // instantiate assembly strategy for mortar elements
  ScaTra::MortarCellAssemblyStrategy strategy(systemmatrix1, matrix1_side_rows, matrix1_side_cols,
      systemmatrix2, matrix2_side_rows, matrix2_side_cols, systemmatrix3, matrix3_side_rows,
      matrix3_side_cols, systemmatrix4, matrix4_side_rows, matrix4_side_cols, systemvector1,
      vector1_side, systemvector2, vector2_side);

  // loop over all mortar elements
  for (int ielement = 0; ielement < ielecolmap.num_my_elements(); ++ielement)
  {
    // extract current mortar element
    auto* const element =
        dynamic_cast<Mortar::Element* const>(idiscret.g_element(ielecolmap.gid(ielement)));
    if (!element) FOUR_C_THROW("Couldn't extract mortar element from mortar discretization!");

    // construct location array for current mortar element
    Core::Elements::LocationArray la(idiscret.num_dof_sets());
    element->location_vector(idiscret, la);

    // initialize element matrices and vectors
    strategy.init_cell_matrices_and_vectors(
        la, la);  // second function argument only serves as dummy

    // evaluate current mortar element
    evaluate_mortar_element(idiscret, *element,
        (Inpar::ScaTra::ImplType)ieleimpltypes.get_local_values()[ielement], la, params,
        strategy.cell_matrix1(), strategy.cell_matrix2(), strategy.cell_matrix3(),
        strategy.cell_matrix4(), strategy.cell_vector1(), strategy.cell_vector2());

    // assemble element matrices and vectors into system matrices and vectors
    strategy.assemble_cell_matrices_and_vectors(la, la, -1);
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
ScaTra::MortarCellInterface* ScaTra::MortarCellFactory::mortar_cell_calc(
    const Inpar::ScaTra::ImplType& impltype, const Mortar::Element& slaveelement,
    const Mortar::Element& masterelement, const Inpar::S2I::CouplingType& couplingtype,
    const Inpar::S2I::InterfaceSides& lmside, const std::string& disname)
{
  // extract number of slave-side degrees of freedom per node
  const int numdofpernode_slave = slaveelement.num_dof_per_node(*slaveelement.nodes()[0]);

  switch (slaveelement.shape())
  {
    case Core::FE::CellType::tri3:
    {
      return mortar_cell_calc<Core::FE::CellType::tri3>(
          impltype, masterelement, couplingtype, lmside, numdofpernode_slave, disname);
      break;
    }

    case Core::FE::CellType::quad4:
    {
      return mortar_cell_calc<Core::FE::CellType::quad4>(
          impltype, masterelement, couplingtype, lmside, numdofpernode_slave, disname);
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid slave-side discretization type!");
      break;
    }
  }

  return nullptr;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s>
ScaTra::MortarCellInterface* ScaTra::MortarCellFactory::mortar_cell_calc(
    const Inpar::ScaTra::ImplType& impltype, const Mortar::Element& masterelement,
    const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
    const int& numdofpernode_slave, const std::string& disname)
{
  // extract number of master-side degrees of freedom per node
  const int numdofpernode_master = masterelement.num_dof_per_node(*masterelement.nodes()[0]);

  switch (masterelement.shape())
  {
    case Core::FE::CellType::tri3:
    {
      return mortar_cell_calc<distype_s, Core::FE::CellType::tri3>(
          impltype, couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    case Core::FE::CellType::quad4:
    {
      return mortar_cell_calc<distype_s, Core::FE::CellType::quad4>(
          impltype, couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid master-side discretization type!");
      break;
    }
  }

  return nullptr;
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellInterface* ScaTra::MortarCellFactory::mortar_cell_calc(
    const Inpar::ScaTra::ImplType& impltype, const Inpar::S2I::CouplingType& couplingtype,
    const Inpar::S2I::InterfaceSides& lmside, const int& numdofpernode_slave,
    const int& numdofpernode_master, const std::string& disname)
{
  // return instance of evaluation class for mortar integration cell depending on physical
  // implementation type
  switch (impltype)
  {
    case Inpar::ScaTra::impltype_std:
    {
      return ScaTra::MortarCellCalc<distype_s, distype_m>::instance(
          couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    case Inpar::ScaTra::impltype_elch_electrode:
    {
      return ScaTra::MortarCellCalcElch<distype_s, distype_m>::instance(
          couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    case Inpar::ScaTra::impltype_elch_electrode_thermo:
    {
      return ScaTra::MortarCellCalcElchSTIThermo<distype_s, distype_m>::instance(
          couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    case Inpar::ScaTra::impltype_thermo_elch_electrode:
    {
      return ScaTra::MortarCellCalcSTIElch<distype_s, distype_m>::instance(
          couplingtype, lmside, numdofpernode_slave, numdofpernode_master, disname);
      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown physical implementation type of mortar integration cell!");
      break;
    }
  }

  return nullptr;
}

/*------------------------------------------------------------------------*
 *------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::init_conv_check_strategy()
{
  if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_petrov or
      couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov)
  {
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyS2ILM>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
  }
  else
    convcheckstrategy_ = std::make_shared<ScaTra::ConvCheckStrategyStd>(
        scatratimint_->scatra_parameter_list()->sublist("NONLINEAR"));
}  // ScaTra::meshtying_strategy_s2_i::init_conv_check_strategy


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::setup_meshtying()
{
  // extract scatra-scatra coupling conditions from discretization
  std::vector<const Core::Conditions::Condition*> s2imeshtying_conditions;
  scatratimint_->discretization()->get_condition("S2IMeshtying", s2imeshtying_conditions);
  std::vector<const Core::Conditions::Condition*> s2ikinetics_conditions;
  scatratimint_->discretization()->get_condition("S2IKinetics", s2ikinetics_conditions);
  kinetics_conditions_meshtying_slaveside_.clear();
  master_conditions_.clear();
  runtime_csvwriter_.emplace(
      Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()),
      scatratimint_->disc_writer()->output(), "kinetics_interface_flux");

  for (auto* s2imeshtying_cond : s2imeshtying_conditions)
  {
    for (auto* s2ikinetics_cond : s2ikinetics_conditions)
    {
      const int s2ikinetics_cond_id = s2ikinetics_cond->parameters().get<int>("ConditionID");
      const int s2ikinetics_cond_interface_side =
          s2ikinetics_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE");

      if (s2ikinetics_cond_id < 0)
        FOUR_C_THROW("Invalid condition ID {} for S2IKinetics Condition!", s2ikinetics_cond_id);

      // only continue if ID's match
      if (s2imeshtying_cond->parameters().get<int>("S2I_KINETICS_ID") != s2ikinetics_cond_id)
        continue;
      // only continue if sides match
      if (s2imeshtying_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") !=
          s2ikinetics_cond_interface_side)
        continue;

      switch (s2ikinetics_cond_interface_side)
      {
        case Inpar::S2I::side_slave:
        {
          if (kinetics_conditions_meshtying_slaveside_.find(s2ikinetics_cond_id) ==
              kinetics_conditions_meshtying_slaveside_.end())
          {
            kinetics_conditions_meshtying_slaveside_.insert(
                std::make_pair(s2ikinetics_cond_id, s2ikinetics_cond));
          }
          else
          {
            FOUR_C_THROW(
                "Cannot have multiple slave-side scatra-scatra interface kinetics conditions with "
                "the same ID {}!",
                s2ikinetics_cond_id);
          }

          if (s2ikinetics_cond->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") ==
              static_cast<int>(Inpar::S2I::kinetics_butlervolmerreducedcapacitance))
          {
            has_capacitive_contributions_ = true;

            auto timeintscheme = Teuchos::getIntegralValue<Inpar::ScaTra::TimeIntegrationScheme>(
                *(scatratimint_->scatra_parameter_list()), "TIMEINTEGR");
            if (not(timeintscheme == Inpar::ScaTra::timeint_bdf2 or
                    timeintscheme == Inpar::ScaTra::timeint_one_step_theta))
            {
              FOUR_C_THROW(
                  "Solution of capacitive interface contributions, i.e. additional transient terms "
                  "is only implemented for OST and BDF2 time integration schemes.");
            }
          }
          std::stringstream condition_flux_name, condition_area_name;
          condition_flux_name << "flux_S2I_condition_" << s2ikinetics_cond_id;
          condition_area_name << "area_S2I_condition_" << s2ikinetics_cond_id;
          runtime_csvwriter_->register_data_vector(condition_flux_name.str(), 1, 16);
          runtime_csvwriter_->register_data_vector(condition_area_name.str(), 1, 16);

          break;
        }

        case Inpar::S2I::side_master:
        {
          if (master_conditions_.find(s2ikinetics_cond_id) == master_conditions_.end())
          {
            master_conditions_.insert(std::make_pair(s2ikinetics_cond_id, s2ikinetics_cond));
          }
          else
          {
            FOUR_C_THROW(
                "Cannot have multiple master-side scatra-scatra interface kinetics conditions with "
                "the same ID {}!",
                s2ikinetics_cond_id);
          }
          break;
        }

        default:
        {
          FOUR_C_THROW("Invalid scatra-scatra interface kinetics condition!");
          break;
        }
      }
    }
  }

  // determine type of mortar meshtying
  switch (couplingtype_)
  {
    // setup scatra-scatra interface coupling for interfaces with pairwise overlapping interface
    // nodes
    case Inpar::S2I::coupling_matching_nodes:
    {
      // overwrite IDs of master-side scatra-scatra interface coupling conditions with the value -1
      // to prevent them from being evaluated when calling evaluate_condition on the discretization
      // TODO: this is somewhat unclean, because changing the conditions, makes calling
      // setup_meshtying() twice invalid (which should not be necessary, but conceptually possible)
      for (auto& mastercondition : master_conditions_)
        const_cast<Core::Conditions::Condition*>(mastercondition.second)
            ->parameters()
            .add("ConditionID", -1);

      if (scatratimint_->num_scal() < 1)
        FOUR_C_THROW("Number of transported scalars not correctly set!");

      // construct new (empty coupling adapter)
      icoup_ = std::make_shared<Coupling::Adapter::Coupling>();

      int num_dof_per_condition = -1;

      // initialize int vectors for global ids of slave and master interface nodes
      if (independent_setup_of_conditions_)
      {
        std::vector<std::vector<int>> islavenodegidvec_cond;
        std::vector<std::vector<int>> imasternodegidvec_cond;

        for (auto& kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
        {
          std::vector<int> islavenodegidvec;
          std::vector<int> imasternodegidvec;

          const int kineticsID = kinetics_slave_cond.first;
          auto* kinetics_condition = kinetics_slave_cond.second;

          if (num_dof_per_condition == -1)
            num_dof_per_condition =
                scatratimint_->num_dof_per_node_in_condition(*kinetics_condition);
          else if (num_dof_per_condition !=
                   scatratimint_->num_dof_per_node_in_condition(*kinetics_condition))
            FOUR_C_THROW("all S2I conditions must have the same number of dof per node");

          if (kinetics_condition->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
              static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
          {
            Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
                *kinetics_condition->get_nodes(), islavenodegidvec);

            auto mastercondition = master_conditions_.find(kineticsID);
            if (mastercondition == master_conditions_.end())
              FOUR_C_THROW("Could not find master condition");

            Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
                *mastercondition->second->get_nodes(), imasternodegidvec);

            islavenodegidvec_cond.push_back(islavenodegidvec);
            imasternodegidvec_cond.push_back(imasternodegidvec);
          }
        }

        icoup_->setup_coupling(*(scatratimint_->discretization()),
            *(scatratimint_->discretization()), imasternodegidvec_cond, islavenodegidvec_cond,
            num_dof_per_condition, true, 1.0e-8);
      }
      else
      {
        std::set<int> islavenodegidset;
        std::set<int> imasternodegidset;

        for (const auto& kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
        {
          const int kineticsID = kinetics_slave_cond.first;
          auto* kinetics_condition = kinetics_slave_cond.second;

          if (num_dof_per_condition == -1)
            num_dof_per_condition =
                scatratimint_->num_dof_per_node_in_condition(*kinetics_condition);
          else if (num_dof_per_condition !=
                   scatratimint_->num_dof_per_node_in_condition(*kinetics_condition))
            FOUR_C_THROW("all S2I conditions must have the same number of dof per node");

          if (kinetics_condition->parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL") !=
              static_cast<int>(Inpar::S2I::kinetics_nointerfaceflux))
          {
            Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
                *kinetics_condition->get_nodes(), islavenodegidset);

            auto mastercondition = master_conditions_.find(kineticsID);
            if (mastercondition == master_conditions_.end())
              FOUR_C_THROW("Could not find master condition");
            else
              Core::Communication::add_owned_node_gid_from_list(*scatratimint_->discretization(),
                  *mastercondition->second->get_nodes(), imasternodegidset);
          }
        }

        std::vector<int> islavenodegidvec(islavenodegidset.begin(), islavenodegidset.end());
        std::vector<int> imasternodegidvec(imasternodegidset.begin(), imasternodegidset.end());

        icoup_->setup_coupling(*(scatratimint_->discretization()),
            *(scatratimint_->discretization()), imasternodegidvec, islavenodegidvec,
            num_dof_per_condition, true, 1.0e-8);
      }

      // generate interior and interface maps
      auto ifullmap = Core::LinAlg::merge_map(icoup_->slave_dof_map(), icoup_->master_dof_map());
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> imaps;
      imaps.emplace_back(
          Core::LinAlg::split_map(*(scatratimint_->discretization()->dof_row_map()), *ifullmap));
      imaps.emplace_back(icoup_->slave_dof_map());
      imaps.emplace_back(icoup_->master_dof_map());

      // initialize global map extractor
      interfacemaps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *(scatratimint_->discretization()->dof_row_map()), imaps);
      interfacemaps_->check_for_valid_map_extractor();

      // initialize interface vector
      // Although the interface vector only contains the transformed master interface dofs, we still
      // initialize it with the full dof_row_map of the discretization to make it work for parallel
      // computations.
      islavephidtnp_ = std::make_shared<Core::LinAlg::Vector<double>>(
          *(scatratimint_->discretization()->dof_row_map()), false);
      imasterphidt_on_slave_side_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
          *(scatratimint_->discretization()->dof_row_map()), false);
      imasterphi_on_slave_side_np_ = std::make_shared<Core::LinAlg::Vector<double>>(
          *(scatratimint_->discretization()->dof_row_map()), false);

      // initialize auxiliary system matrices and associated transformation operators
      islavematrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(icoup_->slave_dof_map()), 81);
      imastermatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*(icoup_->slave_dof_map()), 81);
      imasterslavematrix_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(*(icoup_->slave_dof_map()), 81);
      islavetomasterrowtransform_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
      if (not slaveonly_)
      {
        islavetomastercoltransform_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
        islavetomasterrowcoltransform_ =
            std::make_shared<Coupling::Adapter::MatrixRowColTransform>();
      }

      // initialize auxiliary residual vector
      islaveresidual_ = std::make_shared<Core::LinAlg::Vector<double>>(*(icoup_->slave_dof_map()));

      break;
    }

    // setup scatra-scatra interface coupling for interfaces with non-overlapping interface nodes
    case Inpar::S2I::coupling_mortar_standard:
    case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
    case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
    case Inpar::S2I::coupling_mortar_condensed_petrov:
    case Inpar::S2I::coupling_mortar_condensed_bubnov:
    case Inpar::S2I::coupling_nts_standard:
    {
      // safety checks
      if (imortarredistribution_ and couplingtype_ != Inpar::S2I::coupling_mortar_standard)
      {
        FOUR_C_THROW(
            "Parallel redistribution only implemented for scatra-scatra interface coupling based "
            "on standard mortar approach!");
      }
      if (Teuchos::getIntegralValue<Inpar::Mortar::MeshRelocation>(
              Global::Problem::instance()->mortar_coupling_params(), "MESH_RELOCATION") !=
          Inpar::Mortar::relocation_none)
      {
        FOUR_C_THROW("Mesh relocation not yet implemented for scatra-scatra interface coupling!");
      }

      // initialize empty interface maps
      std::shared_ptr<Core::LinAlg::Map> imastermap =
          std::make_shared<Core::LinAlg::Map>(0, 0, scatratimint_->discretization()->get_comm());
      std::shared_ptr<Core::LinAlg::Map> islavemap =
          std::make_shared<Core::LinAlg::Map>(0, 0, scatratimint_->discretization()->get_comm());
      std::shared_ptr<Core::LinAlg::Map> ifullmap =
          std::make_shared<Core::LinAlg::Map>(0, 0, scatratimint_->discretization()->get_comm());
      if (imortarredistribution_)
      {
        imastermap_ =
            std::make_shared<Core::LinAlg::Map>(0, 0, scatratimint_->discretization()->get_comm());
        islavemap_ =
            std::make_shared<Core::LinAlg::Map>(0, 0, scatratimint_->discretization()->get_comm());
      }

      // loop over all slave-side scatra-scatra interface coupling conditions
      for (auto& kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
      {
        // extract condition ID
        const int condid = kinetics_slave_cond.first;

        // initialize maps for row nodes associated with current condition
        std::map<int, Core::Nodes::Node*> masternodes;
        std::map<int, Core::Nodes::Node*> slavenodes;

        // initialize maps for column nodes associated with current condition
        std::map<int, Core::Nodes::Node*> mastergnodes;
        std::map<int, Core::Nodes::Node*> slavegnodes;

        // initialize maps for elements associated with current condition
        std::map<int, std::shared_ptr<Core::Elements::Element>> masterelements;
        std::map<int, std::shared_ptr<Core::Elements::Element>> slaveelements;

        // extract current slave-side and associated master-side scatra-scatra interface coupling
        // conditions
        std::vector<const Core::Conditions::Condition*> mastercondition(
            1, master_conditions_[condid]);
        std::vector<const Core::Conditions::Condition*> slavecondition(
            1, kinetics_slave_cond.second);

        // fill maps
        Core::Conditions::find_condition_objects(*scatratimint_->discretization(), masternodes,
            mastergnodes, masterelements, mastercondition);
        Core::Conditions::find_condition_objects(*scatratimint_->discretization(), slavenodes,
            slavegnodes, slaveelements, slavecondition);

        // initialize mortar coupling adapter
        icoupmortar_[condid] = std::make_shared<Coupling::Adapter::CouplingMortar>(
            Global::Problem::instance()->n_dim(),
            Global::Problem::instance()->mortar_coupling_params(),
            Global::Problem::instance()->contact_dynamic_params(),
            Global::Problem::instance()->spatial_approximation_type());
        Coupling::Adapter::CouplingMortar& icoupmortar = *icoupmortar_[condid];
        std::vector<int> coupleddof(scatratimint_->num_dof_per_node(), 1);
        icoupmortar.setup_interface(scatratimint_->discretization(),
            scatratimint_->discretization(), coupleddof, mastergnodes, slavegnodes, masterelements,
            slaveelements, scatratimint_->discretization()->get_comm(),
            Global::Problem::instance()->binning_strategy_params(), {},
            Global::Problem::instance()->output_control_file(),
            Global::Problem::instance()->spatial_approximation_type());

        // extract mortar interface
        Mortar::Interface& interface = *icoupmortar.interface();

        // extract mortar discretization
        const Core::FE::Discretization& idiscret = interface.discret();

        if (couplingtype_ != Inpar::S2I::coupling_nts_standard)
        {
          // provide each slave-side mortar element with material of corresponding parent element
          for (int iele = 0; iele < interface.slave_col_elements()->num_my_elements(); ++iele)
          {
            // determine global ID of current slave-side mortar element
            const int elegid = interface.slave_col_elements()->gid(iele);

            // add material
            idiscret.g_element(elegid)->set_material(
                0, Mat::factory(std::dynamic_pointer_cast<Core::Elements::FaceElement>(
                       kinetics_slave_cond.second->geometry().at(elegid))
                           ->parent_element()
                           ->material()
                           ->parameter()
                           ->id()));
          }

          // assign physical implementation type to each slave-side mortar element by copying the
          // physical implementation type of the corresponding parent volume element
          Core::LinAlg::Vector<int> impltypes_row(*interface.slave_row_elements());
          for (int iele = 0; iele < interface.slave_row_elements()->num_my_elements(); ++iele)
          {
            impltypes_row.get_local_values()[iele] =
                dynamic_cast<const Discret::Elements::Transport*>(
                    std::dynamic_pointer_cast<const Core::Elements::FaceElement>(
                        kinetics_slave_cond.second->geometry().at(
                            interface.slave_row_elements()->gid(iele)))
                        ->parent_element())
                    ->impl_type();
          }

          // perform parallel redistribution if desired
          if (imortarredistribution_ and
              Core::Communication::num_mpi_ranks(idiscret.get_comm()) > 1)
          {
            interface.interface_params()
                .sublist("PARALLEL REDISTRIBUTION")
                .set<Inpar::Mortar::ParallelRedist>("PARALLEL_REDIST",
                    Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
                        Global::Problem::instance()->mortar_coupling_params().sublist(
                            "PARALLEL REDISTRIBUTION"),
                        "PARALLEL_REDIST"));
            interface.redistribute();
            interface.fill_complete({}, Global::Problem::instance()->binning_strategy_params(),
                Global::Problem::instance()->output_control_file(),
                Global::Problem::instance()->spatial_approximation_type(), true);
            interface.print_parallel_distribution();
            interface.create_search_tree();
          }

          // generate mortar integration cells
          std::vector<std::shared_ptr<Mortar::IntCell>> imortarcells;
          icoupmortar.evaluate_geometry(imortarcells);

          // assign physical implementation type to each mortar integration cell by copying the
          // physical implementation type of the corresponding slave-side mortar element
          Core::LinAlg::Vector<int> impltypes_col(*interface.slave_col_elements());
          Core::LinAlg::export_to(impltypes_row, impltypes_col);
          imortarcells_[condid].resize(imortarcells.size());
          for (unsigned icell = 0; icell < imortarcells.size(); ++icell)
          {
            imortarcells_[condid][icell] =
                std::pair<std::shared_ptr<Mortar::IntCell>, Inpar::ScaTra::ImplType>(
                    imortarcells[icell],
                    static_cast<Inpar::ScaTra::ImplType>(
                        impltypes_col.get_local_values()[interface.slave_col_elements()->lid(
                            imortarcells[icell]->get_slave_id())]));
          }
        }

        else
        {
          // match slave-side and master-side elements at mortar interface
          switch (interface.search_alg())
          {
            case Inpar::Mortar::search_bfele:
            {
              interface.evaluate_search_brute_force(interface.search_param());
              break;
            }

            case Inpar::Mortar::search_binarytree:
            {
              interface.evaluate_search_binarytree();
              break;
            }

            default:
            {
              FOUR_C_THROW("Invalid search algorithm!");
              break;
            }
          }

          // evaluate normal vectors associated with slave-side nodes
          interface.evaluate_nodal_normals();

          // extract slave-side noderowmap
          const Core::LinAlg::Map& noderowmap_slave = *interface.slave_row_nodes();

          // initialize vector for node-to-segment connectivity, i.e., for pairings between slave
          // nodes and master elements
          std::shared_ptr<Core::LinAlg::Vector<int>>& islavenodestomasterelements =
              islavenodestomasterelements_[condid];
          islavenodestomasterelements =
              std::make_shared<Core::LinAlg::Vector<int>>(noderowmap_slave, false);
          islavenodestomasterelements->put_value(-1);

          // initialize vector for physical implementation types of slave-side nodes
          std::shared_ptr<Core::LinAlg::Vector<int>>& islavenodesimpltypes =
              islavenodesimpltypes_[condid];
          islavenodesimpltypes =
              std::make_shared<Core::LinAlg::Vector<int>>(noderowmap_slave, false);
          islavenodesimpltypes->put_value(Inpar::ScaTra::impltype_undefined);

          // loop over all slave-side nodes
          for (int inode = 0; inode < noderowmap_slave.num_my_elements(); ++inode)
          {
            // extract slave-side node
            auto* const slavenode =
                dynamic_cast<Mortar::Node*>(idiscret.g_node(noderowmap_slave.gid(inode)));
            if (!slavenode)
              FOUR_C_THROW("Couldn't extract slave-side mortar node from mortar discretization!");

            // find associated master-side elements
            std::vector<Mortar::Element*> master_mortar_elements;
            interface.find_master_elements(*slavenode, master_mortar_elements);

            // loop over all master-side elements
            for (auto* master_mortar_ele : master_mortar_elements)
            {
              // extract master-side element
              // project slave-side node onto master-side element
              std::array<double, 2> coordinates_master;
              double dummy(0.);
              Mortar::Projector::impl(*master_mortar_ele)
                  ->project_gauss_point_auxn_3d(slavenode->x().data(), slavenode->mo_data().n(),
                      *master_mortar_ele, coordinates_master.data(), dummy);

              // check whether projected node lies inside master-side element
              if (master_mortar_ele->shape() == Core::FE::CellType::quad4)
              {
                if (coordinates_master[0] < -1. - ntsprojtol_ or
                    coordinates_master[1] < -1. - ntsprojtol_ or
                    coordinates_master[0] > 1. + ntsprojtol_ or
                    coordinates_master[1] > 1. + ntsprojtol_)
                  // projected node lies outside master-side element
                  continue;
              }

              else if (master_mortar_ele->shape() == Core::FE::CellType::tri3)
              {
                if (coordinates_master[0] < -ntsprojtol_ or coordinates_master[1] < -ntsprojtol_ or
                    coordinates_master[0] + coordinates_master[1] > 1. + 2 * ntsprojtol_)
                  // projected node lies outside master-side element
                  continue;
              }

              else
                FOUR_C_THROW("Invalid discretization type of master-side element!");

              // projected node lies inside master-side element
              (*islavenodestomasterelements).get_local_values()[inode] = master_mortar_ele->id();
              break;
            }

            // safety check
            if ((*islavenodestomasterelements).get_local_values()[inode] == -1)
              FOUR_C_THROW("Couldn't match slave-side node with master-side element!");

            // determine physical implementation type of slave-side node based on first associated
            // element
            (*islavenodesimpltypes).get_local_values()[inode] =
                dynamic_cast<Discret::Elements::Transport*>(
                    std::dynamic_pointer_cast<Core::Elements::FaceElement>(
                        kinetics_slave_cond.second->geometry().at(
                            slavenode->adjacent_elements()[0].user_element()->id()))
                        ->parent_element())
                    ->impl_type();
          }

          // extract slave-side elerowmap
          const Core::LinAlg::Map& elecolmap_slave = *interface.slave_col_elements();

          // initialize vector for physical implementation types of slave-side elements
          Core::LinAlg::Vector<int> islaveelementsimpltypes(elecolmap_slave, false);
          islaveelementsimpltypes.put_value(Inpar::ScaTra::impltype_undefined);

          // loop over all slave-side elements
          for (int ielement = 0; ielement < elecolmap_slave.num_my_elements(); ++ielement)
          {
            // determine physical implementation type of current slave-side element
            islaveelementsimpltypes.get_local_values()[ielement] =
                dynamic_cast<Discret::Elements::Transport*>(
                    std::dynamic_pointer_cast<Core::Elements::FaceElement>(
                        kinetics_slave_cond.second->geometry().at(elecolmap_slave.gid(ielement)))
                        ->parent_element())
                    ->impl_type();
          }

          // create parameter list for slave-side elements
          Teuchos::ParameterList eleparams;

          // set action for slave-side elements
          eleparams.set<Inpar::S2I::EvaluationActions>(
              "action", Inpar::S2I::evaluate_nodal_area_fractions);

          // compute vector for lumped interface area fractions associated with slave-side nodes
          const Core::LinAlg::Map& dofrowmap_slave = *interface.slave_row_dofs();
          std::shared_ptr<Core::LinAlg::Vector<double>> islavenodeslumpedareas_dofvector =
              std::make_shared<Core::LinAlg::Vector<double>>(dofrowmap_slave);
          evaluate_mortar_elements(elecolmap_slave, islaveelementsimpltypes, idiscret, eleparams,
              nullptr, Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr,
              Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr,
              Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr,
              Inpar::S2I::side_undefined, Inpar::S2I::side_undefined,
              islavenodeslumpedareas_dofvector != nullptr
                  ? Core::Utils::shared_ptr_from_ref(
                        islavenodeslumpedareas_dofvector->as_multi_vector())
                  : nullptr,
              Inpar::S2I::side_slave, nullptr, Inpar::S2I::side_undefined);

          // transform map of result vector
          std::shared_ptr<Core::LinAlg::Vector<double>>& islavenodeslumpedareas =
              islavenodeslumpedareas_[condid];
          islavenodeslumpedareas = std::make_shared<Core::LinAlg::Vector<double>>(noderowmap_slave);
          for (int inode = 0; inode < noderowmap_slave.num_my_elements(); ++inode)
          {
            (*islavenodeslumpedareas).get_values()[inode] =
                islavenodeslumpedareas_dofvector->local_values_as_span()[dofrowmap_slave.lid(
                    idiscret.dof(idiscret.g_node(noderowmap_slave.gid(inode)), 0))];
          }
        }

        // build interface maps
        imastermap = Core::LinAlg::merge_map(imastermap, interface.master_row_dofs(), false);
        islavemap = Core::LinAlg::merge_map(islavemap, interface.slave_row_dofs(), false);
        ifullmap = Core::LinAlg::merge_map(ifullmap,
            Core::LinAlg::merge_map(interface.master_row_dofs(), interface.slave_row_dofs(), false),
            false);
        if (imortarredistribution_)
        {
          imastermap_ = Core::LinAlg::merge_map(imastermap_, icoupmortar.master_dof_map(), false);
          islavemap_ = Core::LinAlg::merge_map(islavemap_, icoupmortar.slave_dof_map(), false);
        }
      }

      // generate interior and interface maps
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> imaps;
      imaps.emplace_back(
          Core::LinAlg::split_map(*(scatratimint_->discretization()->dof_row_map()), *ifullmap));
      imaps.emplace_back(islavemap);
      imaps.emplace_back(imastermap);

      // initialize global map extractor
      interfacemaps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *(scatratimint_->discretization()->dof_row_map()), imaps);
      interfacemaps_->check_for_valid_map_extractor();

      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_slave or couplingtype_ == Inpar::S2I::coupling_nts_standard)
      {
        // initialize auxiliary system matrix for slave side
        islavematrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*interfacemaps_->map(1), 81);

        // initialize auxiliary residual vector for slave side
        islaveresidual_ = std::make_shared<Core::LinAlg::Vector<double>>(*interfacemaps_->map(1));
      }

      if (couplingtype_ == Inpar::S2I::coupling_mortar_standard or
          lmside_ == Inpar::S2I::side_master or couplingtype_ == Inpar::S2I::coupling_nts_standard)
      {
        // initialize auxiliary system matrix for master side
        imastermatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(
            *interfacemaps_->map(2), 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);

        // initialize auxiliary residual vector for master side
        imasterresidual_ =
            std::make_shared<Core::LinAlg::FEVector<double>>(*interfacemaps_->map(2));
      }

      switch (couplingtype_)
      {
        case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
        case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
        case Inpar::S2I::coupling_mortar_condensed_petrov:
        case Inpar::S2I::coupling_mortar_condensed_bubnov:
        {
          if (lmside_ == Inpar::S2I::side_slave)
          {
            D_ = std::make_shared<Core::LinAlg::SparseMatrix>(*interfacemaps_->map(1), 81);
            M_ = std::make_shared<Core::LinAlg::SparseMatrix>(*interfacemaps_->map(1), 81);
            if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov or
                couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
              E_ = std::make_shared<Core::LinAlg::SparseMatrix>(*interfacemaps_->map(1), 81);
          }
          else
          {
            D_ = std::make_shared<Core::LinAlg::SparseMatrix>(
                *interfacemaps_->map(2), 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
            M_ = std::make_shared<Core::LinAlg::SparseMatrix>(
                *interfacemaps_->map(2), 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
            if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov or
                couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
              E_ = std::make_shared<Core::LinAlg::SparseMatrix>(
                  *interfacemaps_->map(2), 81, true, false, Core::LinAlg::SparseMatrix::FE_MATRIX);
          }

          // loop over all scatra-scatra coupling interfaces
          for (auto& kinetics_slave_cond : kinetics_conditions_meshtying_slaveside_)
          {
            // create parameter list for mortar integration cells
            Teuchos::ParameterList params;

            // add current condition to parameter list
            params.set<const Core::Conditions::Condition*>("condition", kinetics_slave_cond.second);

            // set action
            params.set<Inpar::S2I::EvaluationActions>(
                "action", Inpar::S2I::evaluate_mortar_matrices);

            // evaluate mortar integration cells at current interface
            evaluate_mortar_cells(icoupmortar_[kinetics_slave_cond.first]->interface()->discret(),
                params, D_,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_slave
                                                  : Inpar::S2I::side_master,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_slave
                                                  : Inpar::S2I::side_master,
                M_,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_slave
                                                  : Inpar::S2I::side_master,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_master
                                                  : Inpar::S2I::side_slave,
                E_,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_slave
                                                  : Inpar::S2I::side_master,
                lmside_ == Inpar::S2I::side_slave ? Inpar::S2I::side_slave
                                                  : Inpar::S2I::side_master,
                nullptr, Inpar::S2I::side_undefined, Inpar::S2I::side_undefined, nullptr,
                Inpar::S2I::side_undefined, nullptr, Inpar::S2I::side_undefined);
          }

          // finalize mortar matrices D, M, and E
          D_->complete();
          if (lmside_ == Inpar::S2I::side_slave)
            M_->complete(*interfacemaps_->map(2), *interfacemaps_->map(1));
          else
            M_->complete(*interfacemaps_->map(1), *interfacemaps_->map(2));
          if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov or
              couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
            E_->complete();

          switch (couplingtype_)
          {
            case Inpar::S2I::coupling_mortar_condensed_petrov:
            case Inpar::S2I::coupling_mortar_condensed_bubnov:
            {
              // set up mortar projector P
              std::shared_ptr<Core::LinAlg::Vector<double>> D_diag(nullptr);
              if (lmside_ == Inpar::S2I::side_slave)
                D_diag = std::make_shared<Core::LinAlg::Vector<double>>(*interfacemaps_->map(1));
              else
                D_diag = std::make_shared<Core::LinAlg::Vector<double>>(*interfacemaps_->map(2));
              D_->extract_diagonal_copy(*D_diag);
              D_diag->reciprocal(*D_diag);

              P_ = std::make_shared<Core::LinAlg::SparseMatrix>(*M_);
              P_->left_scale(*D_diag);

              // free memory
              if (!slaveonly_)
              {
                D_ = nullptr;
                M_ = nullptr;
              }

              if (couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
              {
                // set up mortar projector Q
                Q_ = std::make_shared<Core::LinAlg::SparseMatrix>(*E_);
                Q_->left_scale(*D_diag);

                // free memory
                E_ = nullptr;
              }

              break;
            }

            case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
            case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
            {
              // determine number of Lagrange multiplier dofs owned by each processor
              MPI_Comm comm(scatratimint_->discretization()->get_comm());
              const int numproc(Core::Communication::num_mpi_ranks(comm));
              const int mypid(Core::Communication::my_mpi_rank(comm));
              std::vector<int> localnumlmdof(numproc, 0);
              std::vector<int> globalnumlmdof(numproc, 0);
              if (lmside_ == Inpar::S2I::side_slave)
                localnumlmdof[mypid] = interfacemaps_->map(1)->num_my_elements();
              else
                localnumlmdof[mypid] = interfacemaps_->map(2)->num_my_elements();
              globalnumlmdof = Core::Communication::sum_all(localnumlmdof, comm);

              // for each processor, determine offset of minimum Lagrange multiplier dof GID w.r.t.
              // maximum standard dof GID
              int offset(0);
              for (int ipreviousproc = 0; ipreviousproc < mypid; ++ipreviousproc)
                offset += globalnumlmdof[ipreviousproc];

              // for each processor, determine Lagrange multiplier dof GIDs
              std::vector<int> lmdofgids(globalnumlmdof[mypid], 0);
              for (int lmdoflid = 0; lmdoflid < globalnumlmdof[mypid]; ++lmdoflid)
                lmdofgids[lmdoflid] =
                    scatratimint_->dof_row_map()->max_all_gid() + 1 + offset + lmdoflid;

              // build Lagrange multiplier dofrowmap
              const std::shared_ptr<Core::LinAlg::Map> lmdofrowmap =
                  std::make_shared<Core::LinAlg::Map>(
                      -1, (int)lmdofgids.size(), lmdofgids.data(), 0, comm);

              // initialize vectors associated with Lagrange multiplier dofs
              lm_ = std::make_shared<Core::LinAlg::Vector<double>>(*lmdofrowmap);
              lmresidual_ = std::make_shared<Core::LinAlg::Vector<double>>(*lmdofrowmap);
              lmincrement_ = std::make_shared<Core::LinAlg::Vector<double>>(*lmdofrowmap);

              // initialize extended map extractor
              std::shared_ptr<Core::LinAlg::Map> extendedmap = Core::LinAlg::merge_map(
                  *(scatratimint_->discretization()->dof_row_map()), *lmdofrowmap, false);
              extendedmaps_ = std::make_shared<Core::LinAlg::MapExtractor>(
                  *extendedmap, lmdofrowmap, scatratimint_->discretization()->dof_row_map());
              extendedmaps_->check_for_valid_map_extractor();

              // transform range map of mortar matrices D and M from slave-side dofrowmap to
              // Lagrange multiplier dofrowmap
              D_ = Core::LinAlg::matrix_row_transform_gids(*D_, *lmdofrowmap);
              M_ = Core::LinAlg::matrix_row_transform_gids(*M_, *lmdofrowmap);

              if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_petrov)
              {
                // transform domain map of mortar matrix D from slave-side dofrowmap to Lagrange
                // multiplier dofrowmap and store transformed matrix as mortar matrix E
                E_ = Core::LinAlg::matrix_col_transform_gids(*D_, *lmdofrowmap);
              }
              else
              {
                // transform domain and range maps of mortar matrix E from slave-side dofrowmap to
                // Lagrange multiplier dofrowmap
                E_ = Core::LinAlg::matrix_row_col_transform_gids(*E_, *lmdofrowmap, *lmdofrowmap);
              }

              break;
            }

            default:
            {
              FOUR_C_THROW("Invalid type of mortar meshtying!");
              break;
            }
          }

          break;
        }

        default:
        {
          // do nothing
          break;
        }
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Type of mortar meshtying for scatra-scatra interface coupling not recognized!");
      break;
    }
  }

  // further initializations depending on type of global system matrix
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::sparse:
    {
      // nothing needs to be done in this case
      break;
    }
    case Core::LinAlg::MatrixType::block_condition:
    case Core::LinAlg::MatrixType::block_condition_dof:
    {
      // safety check
      const bool allowed_block_system_solvers =
          scatratimint_->solver()->params().isSublist("AMGnxn Parameters") or
          scatratimint_->solver()->params().isSublist("Teko Parameters") or
          scatratimint_->solver()->params().isSublist("MueLu Parameters");
      FOUR_C_ASSERT_ALWAYS(allowed_block_system_solvers,
          "Global system matrix with block structure requires AMGnxn, MueLu or Teko block "
          "preconditioner!");

      // initialize map extractors associated with blocks of global system matrix
      build_block_map_extractors();

      break;
    }
    default:
    {
      FOUR_C_THROW(
          "{} is not a valid 'ScaTra::MatrixType'. Set a valid 'ScaTra::MatrixType' in your input "
          "file!",
          static_cast<int>(matrixtype_));
      break;
    }
  }

  // setup evaluation of scatra-scatra interface layer growth if applicable
  if (scatratimint_->discretization()->has_condition("S2IKineticsGrowth"))
  {
    std::vector<const Core::Conditions::Condition*> growth_conditions;
    scatratimint_->discretization()->get_condition("S2IKineticsGrowth", growth_conditions);
    FOUR_C_ASSERT_ALWAYS(growth_conditions.size() == 1, "Expected only one growth condition!");
    const auto* condition = growth_conditions[0];
    // perform setup depending on evaluation method
    switch (intlayergrowth_evaluation_)
    {
      case Inpar::S2I::growth_evaluation_monolithic:
      case Inpar::S2I::growth_evaluation_semi_implicit:
      {
        // extract map associated with scatra-scatra interface layer thicknesses
        const std::shared_ptr<const Core::LinAlg::Map>& dofrowmap_growth =
            scatratimint_->dof_row_map(2);

        // initialize state vector of discrete scatra-scatra interface layer thicknesses at time n
        growthn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);

        // additional initializations for monolithic solution approach
        if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
        {
          // initialize extended map extractor
          const Core::LinAlg::Map* const dofrowmap_scatra =
              scatratimint_->discretization()->dof_row_map();
          extendedmaps_ = std::make_shared<Core::LinAlg::MapExtractor>(
              *Core::LinAlg::merge_map(*dofrowmap_scatra, *dofrowmap_growth, false),
              scatratimint_->dof_row_map(2), dofrowmap_scatra);
          extendedmaps_->check_for_valid_map_extractor();

          // initialize additional state vectors
          growthnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);
          growthdtn_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);
          growthdtnp_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);
          growthhist_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);
          growthresidual_ = std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);
          growthincrement_ =
              std::make_shared<Core::LinAlg::Vector<double>>(*dofrowmap_growth, true);

          // initialize map extractors and global matrix blocks
          growthgrowthblock_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap_growth, 81);
          switch (matrixtype_)
          {
            case Core::LinAlg::MatrixType::sparse:
            {
              // initialize extended map extractor associated with blocks of global system matrix
              extendedblockmaps_ = extendedmaps_;

              scatragrowthblock_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap_scatra,
                  81);  // We actually don't really need the entire scalar transport dofrowmap
                        // here, but only a submap associated with all (slave-side and master-side)
                        // interfacial degrees of freedom. However, this will later cause an error
                        // in debug mode when assigning the scatra-growth matrix block to the
                        // global system matrix in the Solve() routine.
              growthscatrablock_ =
                  std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap_growth, 81);

              break;
            }

            case Core::LinAlg::MatrixType::block_condition:
            case Core::LinAlg::MatrixType::block_condition_dof:
            {
              // initialize map extractor associated with all degrees of freedom for scatra-scatra
              // interface layer growth
              blockmapgrowth_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(*dofrowmap_growth,
                  std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, dofrowmap_growth));
              blockmapgrowth_->check_for_valid_map_extractor();

              // initialize extended map extractor associated with blocks of global system matrix
              const unsigned nblockmaps = scatratimint_->dof_block_maps()->num_maps();
              std::vector<std::shared_ptr<const Core::LinAlg::Map>> extendedblockmaps(
                  nblockmaps + 1, nullptr);
              for (int iblockmap = 0; iblockmap < static_cast<int>(nblockmaps); ++iblockmap)
                extendedblockmaps[iblockmap] = scatratimint_->dof_block_maps()->map(iblockmap);
              extendedblockmaps[nblockmaps] = dofrowmap_growth;
              extendedblockmaps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
                  *extendedmaps_->full_map(), extendedblockmaps);
              extendedblockmaps_->check_for_valid_map_extractor();

              scatragrowthblock_ = std::make_shared<
                  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                  *blockmapgrowth_, *scatratimint_->dof_block_maps(), 81, false, true);
              growthscatrablock_ = std::make_shared<
                  Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
                  *scatratimint_->dof_block_maps(), *blockmapgrowth_, 81, false, true);

              break;
            }

            default:
            {
              FOUR_C_THROW(
                  "Type of global system matrix for scatra-scatra interface coupling involving "
                  "interface layer growth not recognized!");
            }
          }

          // initialize extended system matrix including rows and columns associated with
          // scatra-scatra interface layer thickness variables
          extendedsystemmatrix_ = std::make_shared<
              Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(

              *extendedblockmaps_, *extendedblockmaps_);
        }

        // loop over all boundary conditions for scatra-scatra interface coupling
        for (auto& icond : s2ikinetics_conditions)
        {
          // check whether current boundary condition is associated with boundary condition for
          // scatra-scatra interface layer growth
          if (icond->parameters().get<int>("ConditionID") ==
              condition->parameters().get<int>("ConditionID"))
            // copy conductivity parameter
            const_cast<Core::Conditions::Condition*>(icond)->parameters().add(
                "CONDUCTIVITY", condition->parameters().get<double>("CONDUCTIVITY"));
        }

        // extract initial scatra-scatra interface layer thickness from condition
        const double initthickness = condition->parameters().get<double>("INITTHICKNESS");

        // extract nodal cloud from condition
        const std::vector<int>* nodegids = condition->get_nodes();

        // loop over all nodes
        for (int nodegid : *nodegids)
        {
          // extract global ID of current node
          // process only nodes stored by current processor
          if (scatratimint_->discretization()->have_global_node(nodegid))
          {
            // extract current node
            const Core::Nodes::Node* const node = scatratimint_->discretization()->g_node(nodegid);

            // process only nodes owned by current processor
            if (node->owner() ==
                Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()))
            {
              // extract local ID of scatra-scatra interface layer thickness variable associated
              // with current node
              const int doflid_growth = scatratimint_->discretization()->dof_row_map(2)->lid(
                  scatratimint_->discretization()->dof(2, node, 0));
              if (doflid_growth < 0)
              {
                FOUR_C_THROW(
                    "Couldn't extract local ID of scatra-scatra interface layer thickness "
                    "variable!");
              }

              // set initial value
              (*growthn_).get_values()[doflid_growth] = initthickness;
            }  // nodes owned by current processor
          }  // nodes stored by current processor
        }  // loop over all nodes

        // copy initial state
        if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
          growthnp_->update(1., *growthn_, 0.);

        break;
      }

      default:
      {
        FOUR_C_THROW(
            "Unknown evaluation method for scatra-scatra interface coupling involving interface "
            "layer growth!");
        break;
      }
    }
  }

  // instantiate appropriate equilibration class
  auto equilibration_method =
      std::vector<Core::LinAlg::EquilibrationMethod>(1, scatratimint_->equilibration_method());
  equilibration_ = Core::LinAlg::build_equilibration(matrixtype_, equilibration_method,
      (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic
              ? extendedmaps_->full_map()
              : std::make_shared<const Core::LinAlg::Map>(
                    *scatratimint_->discretization()->dof_row_map())));
}  // ScaTra::meshtying_strategy_s2_i::setup_meshtying

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::compute_time_derivative() const
{
  // only relevant for monolithic evaluation of scatra-scatra interface layer growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
  {
    // compute inverse time factor 1./(theta*dt)
    const double timefac_inverse =
        1. / scatratimint_->scatra_time_parameter_list()->get<double>("time factor");

    // compute state vector of time derivatives of discrete scatra-scatra interface layer
    // thicknesses
    growthdtnp_->update(timefac_inverse, *growthnp_, -timefac_inverse, *growthhist_, 0.);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::update() const
{
  // only relevant for monolithic evaluation of scatra-scatra interface layer growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
  {
    // update state vectors
    growthn_->update(1., *growthnp_, 0.);
    growthdtn_->update(1., *growthdtnp_, 0.);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::set_element_general_parameters(
    Teuchos::ParameterList& parameters) const
{
  // add local Newton-Raphson convergence tolerance for scatra-scatra interface layer growth to
  // parameter list
  parameters.set<double>("intlayergrowth_convtol", intlayergrowth_convtol_);

  // add maximum number of local Newton-Raphson iterations for scatra-scatra interface layer growth
  // to parameter list
  parameters.set<unsigned>("intlayergrowth_itemax", intlayergrowth_itemax_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::set_condition_specific_scatra_parameters(
    const Core::Conditions::Condition& s2icondition) const
{
  Teuchos::ParameterList conditionparams;

  // fill the parameter list
  write_s2_i_kinetics_specific_scatra_parameters_to_parameter_list(s2icondition, conditionparams);

  Discret::Elements::ScaTraEleParameterBoundary::instance("scatra")->set_parameters(
      conditionparams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::write_s2_i_kinetics_specific_scatra_parameters_to_parameter_list(
    const Core::Conditions::Condition& s2ikinetics_cond,
    Teuchos::ParameterList& s2icouplingparameters)
{
  // get kinetic model and condition type
  const Core::Conditions::ConditionType conditiontype = s2ikinetics_cond.type();

  // set kinetic model, condition type and numscal
  s2icouplingparameters.set<Core::Conditions::ConditionType>("condition type", conditiontype);

  // set the condition type specific parameters
  switch (conditiontype)
  {
    case Core::Conditions::ConditionType::S2IKinetics:
    {
      const int kineticmodel =
          s2ikinetics_cond.parameters().get<Inpar::S2I::KineticModels>("KINETIC_MODEL");
      s2icouplingparameters.set("KINETIC_MODEL", kineticmodel);

      // set the kinetic model specific parameters
      switch (kineticmodel)
      {
        case Inpar::S2I::kinetics_constperm:
        case Inpar::S2I::kinetics_linearperm:
        {
          s2icouplingparameters.set<int>(
              "NUMSCAL", s2ikinetics_cond.parameters().get<int>("NUMSCAL"));
          s2icouplingparameters.set<const std::vector<double>*>("PERMEABILITIES",
              &s2ikinetics_cond.parameters().get<std::vector<double>>("PERMEABILITIES"));
          s2icouplingparameters.set<bool>(
              "IS_PSEUDO_CONTACT", s2ikinetics_cond.parameters().get<bool>("IS_PSEUDO_CONTACT"));
          break;
        }

        case Inpar::S2I::kinetics_constantinterfaceresistance:
        {
          s2icouplingparameters.set<double>(
              "RESISTANCE", s2ikinetics_cond.parameters().get<double>("RESISTANCE"));
          s2icouplingparameters.set<const std::vector<int>*>(
              "ONOFF", &s2ikinetics_cond.parameters().get<std::vector<int>>("ONOFF"));
          s2icouplingparameters.set<int>(
              "numelectrons", s2ikinetics_cond.parameters().get<int>("E-"));
          s2icouplingparameters.set<bool>(
              "IS_PSEUDO_CONTACT", s2ikinetics_cond.parameters().get<bool>("IS_PSEUDO_CONTACT"));
          break;
        }

        case Inpar::S2I::kinetics_nointerfaceflux:
        {
          // do nothing
          break;
        }

        case Inpar::S2I::kinetics_butlervolmer:
        case Inpar::S2I::kinetics_butlervolmerlinearized:
        case Inpar::S2I::kinetics_butlervolmerreduced:
        case Inpar::S2I::kinetics_butlervolmerreducedcapacitance:
        case Inpar::S2I::kinetics_butlervolmerreducedlinearized:
        case Inpar::S2I::kinetics_butlervolmerpeltier:
        case Inpar::S2I::kinetics_butlervolmerresistance:
        case Inpar::S2I::kinetics_butlervolmerreducedthermoresistance:
        case Inpar::S2I::kinetics_butlervolmerreducedresistance:
        {
          s2icouplingparameters.set<int>(
              "NUMSCAL", s2ikinetics_cond.parameters().get<int>("NUMSCAL"));
          s2icouplingparameters.set<const std::vector<int>*>("STOICHIOMETRIES",
              &s2ikinetics_cond.parameters().get<std::vector<int>>("STOICHIOMETRIES"));
          s2icouplingparameters.set<int>(
              "numelectrons", s2ikinetics_cond.parameters().get<int>("E-"));
          s2icouplingparameters.set<double>(
              "K_R", s2ikinetics_cond.parameters().get<double>("K_R"));
          s2icouplingparameters.set<double>(
              "ALPHA_A", s2ikinetics_cond.parameters().get<double>("ALPHA_A"));
          s2icouplingparameters.set<double>(
              "ALPHA_C", s2ikinetics_cond.parameters().get<double>("ALPHA_C"));
          s2icouplingparameters.set<bool>(
              "IS_PSEUDO_CONTACT", s2ikinetics_cond.parameters().get<bool>("IS_PSEUDO_CONTACT"));

          if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedcapacitance)
            s2icouplingparameters.set<double>(
                "CAPACITANCE", s2ikinetics_cond.parameters().get<double>("CAPACITANCE"));

          if (kineticmodel == Inpar::S2I::kinetics_butlervolmerpeltier)
            s2icouplingparameters.set<double>(
                "PELTIER", s2ikinetics_cond.parameters().get<double>("PELTIER"));

          if (kineticmodel == Inpar::S2I::kinetics_butlervolmerresistance or
              kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedresistance)
          {
            s2icouplingparameters.set<double>(
                "RESISTANCE", s2ikinetics_cond.parameters().get<double>("RESISTANCE"));
            s2icouplingparameters.set<double>("CONVTOL_IMPLBUTLERVOLMER",
                s2ikinetics_cond.parameters().get<double>("CONVTOL_IMPLBUTLERVOLMER"));
            s2icouplingparameters.set<int>("ITEMAX_IMPLBUTLERVOLMER",
                s2ikinetics_cond.parameters().get<int>("ITEMAX_IMPLBUTLERVOLMER"));
          }

          if (kineticmodel == Inpar::S2I::kinetics_butlervolmerreducedthermoresistance)
          {
            s2icouplingparameters.set<double>(
                "THERMOPERM", s2ikinetics_cond.parameters().get<double>("THERMOPERM"));
            s2icouplingparameters.set<double>("MOLAR_HEAT_CAPACITY",
                s2ikinetics_cond.parameters().get<double>("MOLAR_HEAT_CAPACITY"));
          }
          break;
        }

        default:
        {
          FOUR_C_THROW("Not implemented for this kinetic model: {}", kineticmodel);
        }
      }
      break;
    }

    case Core::Conditions::ConditionType::S2IKineticsGrowth:
    {
      const int kineticmodel =
          s2ikinetics_cond.parameters().get<Inpar::S2I::GrowthKineticModels>("KINETIC_MODEL");
      s2icouplingparameters.set("KINETIC_MODEL", kineticmodel);

      // set the kinetic model specific parameters
      switch (kineticmodel)
      {
        case Inpar::S2I::growth_kinetics_butlervolmer:
        {
          s2icouplingparameters.set<int>(
              "NUMSCAL", s2ikinetics_cond.parameters().get<int>("NUMSCAL"));
          s2icouplingparameters.set<const std::vector<int>*>("STOICHIOMETRIES",
              &s2ikinetics_cond.parameters().get<std::vector<int>>("STOICHIOMETRIES"));
          s2icouplingparameters.set<int>(
              "numelectrons", s2ikinetics_cond.parameters().get<int>("E-"));
          s2icouplingparameters.set<double>(
              "K_R", s2ikinetics_cond.parameters().get<double>("K_R"));
          s2icouplingparameters.set<double>(
              "ALPHA_A", s2ikinetics_cond.parameters().get<double>("ALPHA_A"));
          s2icouplingparameters.set<double>(
              "ALPHA_C", s2ikinetics_cond.parameters().get<double>("ALPHA_C"));
          s2icouplingparameters.set<double>(
              "DENSITY", s2ikinetics_cond.parameters().get<double>("DENSITY"));
          s2icouplingparameters.set<double>(
              "MOLMASS", s2ikinetics_cond.parameters().get<double>("MOLMASS"));
          s2icouplingparameters.set<double>(
              "REGPAR", s2ikinetics_cond.parameters().get<double>("REGPAR"));
          s2icouplingparameters.set("REGTYPE",
              s2ikinetics_cond.parameters().get<Inpar::S2I::RegularizationType>("REGTYPE"));
          s2icouplingparameters.set<double>(
              "CONDUCTIVITY", s2ikinetics_cond.parameters().get<double>("CONDUCTIVITY"));
          break;
        }

        default:
        {
          FOUR_C_THROW("Not implemented for this kinetic model: {}", kineticmodel);
        }
      }
      break;
    }

    default:
    {
      FOUR_C_THROW("Not implemented for this condition type: {}", conditiontype);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::set_old_part_of_rhs() const
{
  // only relevant for monolithic evaluation of scatra-scatra interface layer growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
  {
    // compute factor dt*(1-theta)
    const double factor = scatratimint_->dt() -
                          scatratimint_->scatra_time_parameter_list()->get<double>("time factor");

    // compute history vector
    growthhist_->update(1., *growthn_, factor, *growthdtn_, 0.);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::write_restart() const
{
  // only relevant for monolithic or semi-implicit evaluation of scatra-scatra interface layer
  // growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic or
      intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit)
  {
    // output state vector of discrete scatra-scatra interface layer thicknesses
    scatratimint_->disc_writer()->write_vector("growthn", growthn_);

    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
    {
      // output state vector of time derivatives of discrete scatra-scatra interface layer
      // thicknesses
      scatratimint_->disc_writer()->write_vector("growthdtn", growthdtn_);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::read_restart(
    const int step, std::shared_ptr<Core::IO::InputControl> input) const
{
  // only relevant for monolithic or semi-implicit evaluation of scatra-scatra interface layer
  // growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic or
      intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit)
  {
    // initialize reader
    std::shared_ptr<Core::IO::DiscretizationReader> reader(nullptr);
    if (input == nullptr)
      reader = std::make_shared<Core::IO::DiscretizationReader>(*scatratimint_->discretization(),
          Global::Problem::instance()->input_control_file(), step);
    else
      reader = std::make_shared<Core::IO::DiscretizationReader>(
          *scatratimint_->discretization(), input, step);

    // read state vector of discrete scatra-scatra interface layer thicknesses
    reader->read_vector(growthn_, "growthn");

    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
    {
      // read state vector of time derivatives of discrete scatra-scatra interface layer thicknesses
      reader->read_vector(growthdtn_, "growthdtn");

      // copy restart state
      growthnp_->update(1., *growthn_, 0.);
      growthdtnp_->update(1., *growthdtn_, 0.);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::FE::Discretization& ScaTra::MeshtyingStrategyS2I::mortar_discretization(
    const int& condid) const
{
  return icoupmortar_.at(condid)->interface()->discret();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::collect_output_data() const
{
  // only relevant for monolithic or semi-implicit evaluation of scatra-scatra interface layer
  // growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic or
      intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit)
  {
    // extract relevant state vector of discrete scatra-scatra interface layer thicknesses based on
    // map of scatra-scatra interface layer thickness variables
    const Core::LinAlg::Vector<double>& growth =
        intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic ? *growthnp_
                                                                               : *growthn_;

    // for proper output, initialize target state vector of discrete scatra-scatra interface layer
    // thicknesses based on map of row nodes
    auto intlayerthickness =
        Core::LinAlg::Vector<double>(*scatratimint_->discretization()->node_row_map(), true);

    // extract boundary condition for scatra-scatra interface layer growth
    std::vector<const Core::Conditions::Condition*> conds;
    scatratimint_->discretization()->get_condition("S2IKineticsGrowth", conds);
    FOUR_C_ASSERT_ALWAYS(conds.size() == 1, "Required exactly one S2IKineticsGrowth condition!");

    // extract nodal cloud from condition
    const std::vector<int>* nodegids = conds.front()->get_nodes();

    // loop over all nodes
    for (int nodegid : *nodegids)
    {
      // extract global ID of current node
      // process only nodes stored by current processor
      if (scatratimint_->discretization()->have_global_node(nodegid))
      {
        // extract current node
        const Core::Nodes::Node* const node = scatratimint_->discretization()->g_node(nodegid);

        // process only nodes owned by current processor
        if (node->owner() ==
            Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()))
        {
          // extract local ID of current node
          const int nodelid = scatratimint_->discretization()->node_row_map()->lid(nodegid);
          if (nodelid < 0) FOUR_C_THROW("Couldn't extract local node ID!");

          // extract local ID of scatra-scatra interface layer thickness variable associated with
          // current node
          const int doflid_growth = scatratimint_->discretization()->dof_row_map(2)->lid(
              scatratimint_->discretization()->dof(2, node, 0));
          if (doflid_growth < 0)
            FOUR_C_THROW(
                "Couldn't extract local ID of scatra-scatra interface layer thickness variable!");

          // copy thickness variable into target state vector of discrete scatra-scatra interface
          // layer thicknesses
          (intlayerthickness).get_values()[nodelid] = growth.local_values_as_span()[doflid_growth];
        }  // nodes owned by current processor
      }  // nodes stored by current processor
    }  // loop over all nodes

    // output target state vector of discrete scatra-scatra interface layer thicknesses
    scatratimint_->visualization_writer().append_result_data_vector_with_context(
        intlayerthickness, Core::IO::OutputEntity::node, {"intlayerthickness"});
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::output() const
{
  if (output_interface_flux_) output_interface_flux();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::output_interface_flux() const
{
  add_time_integration_specific_vectors();
  std::vector<const Core::Conditions::Condition*> s2ikinetics_conditions;
  scatratimint_->discretization()->get_condition("S2IKinetics", s2ikinetics_conditions);

  std::map<std::string, std::vector<double>> output_data;

  for (auto* s2ikinetics_cond : s2ikinetics_conditions)
  {
    // only slave side has relevant information
    if (s2ikinetics_cond->parameters().get<Inpar::S2I::InterfaceSides>("INTERFACE_SIDE") ==
        static_cast<int>(Inpar::S2I::side_slave))
    {
      const int condition_id = s2ikinetics_cond->parameters().get<int>("ConditionID");
      auto s2i_flux =
          std::make_shared<Core::LinAlg::SerialDenseVector>(scatratimint_->num_dof_per_node());
      Core::LinAlg::SerialDenseVector boundaryint_vector(1);
      {
        Teuchos::ParameterList condparams;

        Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
            "action", ScaTra::BoundaryAction::calc_s2icoupling_flux, condparams);

        scatratimint_->discretization()->evaluate_scalars(
            condparams, *s2i_flux, "S2IKinetics", condition_id);
      }

      {
        Teuchos::ParameterList condparams;

        // overwrite action in parameter list
        Core::Utils::add_enum_class_to_parameter_list<ScaTra::BoundaryAction>(
            "action", ScaTra::BoundaryAction::calc_boundary_integral, condparams);

        // compute value of boundary integral
        scatratimint_->discretization()->evaluate_scalars(
            condparams, boundaryint_vector, "S2IKinetics", condition_id);
      }

      // extract value of boundary integral
      const double boundaryint = (boundaryint_vector)(0);
      s2i_flux->scale(1.0 / boundaryint);

      std::stringstream condition_flux_name, condition_area_name;
      condition_flux_name << "flux_S2I_condition_" << condition_id;
      condition_area_name << "area_S2I_condition_" << condition_id;

      FOUR_C_ASSERT(
          runtime_csvwriter_.has_value(), "internal error: runtime csv writer not created.");
      output_data[condition_flux_name.str()] = {(*s2i_flux)[1]};
      output_data[condition_area_name.str()] = {boundaryint};
    }
  }
  runtime_csvwriter_->write_data_to_file(scatratimint_->time(), scatratimint_->step(), output_data);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::explicit_predictor() const
{
  // only relevant for monolithic evaluation of scatra-scatra interface layer growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
    // predict state vector of discrete scatra-scatra interface layer thicknesses at time n+1
    growthnp_->update(scatratimint_->dt(), *growthdtn_, 1.);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::extract_matrix_rows(
    const Core::LinAlg::SparseMatrix& matrix,  //!< source matrix
    Core::LinAlg::SparseMatrix& rows,          //!< destination matrix
    const Core::LinAlg::Map& rowmap            //!< map of matrix rows to be extracted
)
{
  // safety check
  if (rows.filled())
    FOUR_C_THROW("Source matrix rows cannot be extracted into filled destination matrix!");

  // loop over all source matrix rows to be extracted
  for (int doflid = 0; doflid < rowmap.num_my_elements(); ++doflid)
  {
    // determine global ID of current matrix row
    const int dofgid = rowmap.gid(doflid);
    if (dofgid < 0) FOUR_C_THROW("Couldn't find local ID {} in map!", doflid);

    // extract current matrix row from source matrix
    const int length = matrix.num_global_entries(dofgid);
    int numentries(0);
    std::vector<double> values(length, 0.);
    std::vector<int> indices(length, 0);
    matrix.extract_global_row_copy(dofgid, length, numentries, values.data(), indices.data());

    // copy current source matrix row into destination matrix
    rows.insert_global_values(dofgid, numentries, values.data(), indices.data());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::add_time_integration_specific_vectors() const
{
  // only relevant for scatra-scatra interface coupling with pairwise coinciding interface nodes
  if (couplingtype_ == Inpar::S2I::coupling_matching_nodes)
  {
    // add state vector containing master-side scatra degrees of freedom to scatra discretization
    interfacemaps_->insert_vector(
        *icoup_->master_to_slave(*interfacemaps_->extract_vector(*(scatratimint_->phiafnp()), 2)),
        1, *imasterphi_on_slave_side_np_);
    scatratimint_->discretization()->set_state("imasterphinp", *imasterphi_on_slave_side_np_);

    if (has_capacitive_contributions_)
    {
      interfacemaps_->insert_vector(
          *interfacemaps_->extract_vector(*(scatratimint_->phidtnp()), 1), 1, *islavephidtnp_);
      scatratimint_->discretization()->set_state("islavephidtnp", *islavephidtnp_);
      interfacemaps_->insert_vector(
          *icoup_->master_to_slave(*interfacemaps_->extract_vector(*(scatratimint_->phidtnp()), 2)),
          1, *imasterphidt_on_slave_side_np_);
      scatratimint_->discretization()->set_state("imasterphidtnp", *imasterphidt_on_slave_side_np_);
    }
  }

  // only relevant for monolithic or semi-implicit evaluation of scatra-scatra interface layer
  // growth
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic or
      intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit)
  {
    // extract relevant state vector of discrete scatra-scatra interface layer thicknesses
    const std::shared_ptr<Core::LinAlg::Vector<double>>& growth =
        intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic ? growthnp_
                                                                               : growthn_;

    // set state vector
    scatratimint_->discretization()->set_state(2, "growth", *growth);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::compute_time_step_size(double& dt)
{
  // not implemented for standard scalar transport
  if (intlayergrowth_timestep_ > 0.)
  {
    FOUR_C_THROW(
        "Adaptive time stepping for scatra-scatra interface layer growth not implemented for "
        "standard scalar transport!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::init_meshtying()
{
  // instantiate strategy for Newton-Raphson convergence check
  init_conv_check_strategy();

  // extract boundary conditions for scatra-scatra interface layer growth
  std::vector<const Core::Conditions::Condition*> conditions;
  scatratimint_->discretization()->get_condition("S2IKineticsGrowth", conditions);

  // initialize scatra-scatra interface layer growth
  if (conditions.size())
  {
    // safety checks
    if (conditions.size() != 1)
    {
      FOUR_C_THROW(
          "Can't have more than one boundary condition for scatra-scatra interface layer growth at "
          "the moment!");
    }
    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_none)
    {
      FOUR_C_THROW(
          "Invalid flag for evaluation of scatra-scatra interface coupling involving interface "
          "layer growth!");
    }
    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic and
        scatratimint_->method_name() != Inpar::ScaTra::timeint_one_step_theta)
    {
      FOUR_C_THROW(
          "Monolithic evaluation of scatra-scatra interface layer growth only implemented for "
          "one-step-theta time integration scheme at the moment!");
    }
    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_semi_implicit and
        conditions[0]->parameters().get<Inpar::S2I::RegularizationType>("REGTYPE") !=
            Inpar::S2I::RegularizationType::regularization_none)
    {
      FOUR_C_THROW(
          "No regularization implemented for semi-implicit evaluation of scatra-scatra interface "
          "layer growth!");
    }
    if (couplingtype_ != Inpar::S2I::coupling_matching_nodes)
    {
      FOUR_C_THROW(
          "Evaluation of scatra-scatra interface layer growth only implemented for conforming "
          "interface discretizations!");
    }

    // provide scalar transport discretization with additional dofset for scatra-scatra interface
    // layer thickness
    const auto& col_map = *scatratimint_->discretization()->node_col_map();
    const auto numdofpernode = std::make_shared<Core::LinAlg::Vector<int>>(col_map);
    const auto conditioned_node_ids =
        Core::Conditions::find_conditioned_node_ids(*scatratimint_->discretization(),
            "S2IKineticsGrowth", Core::Conditions::LookFor::locally_owned_and_ghosted);
    const std::span<const int> my_col_nodes(
        col_map.my_global_elements(), col_map.num_my_elements());
    for (const int& inode : my_col_nodes)
    {
      if (conditioned_node_ids.contains(inode))
        // add one degree of freedom for scatra-scatra interface layer thickness to current node
        (*numdofpernode).get_local_values()[inode] = 1;
    }

    int number_dofsets = scatratimint_->get_max_dof_set_number();
    std::shared_ptr<Core::DOFSets::DofSetInterface> dofset =
        std::make_shared<Core::DOFSets::DofSetPredefinedDoFNumber>(
            numdofpernode, nullptr, nullptr, true);
    if (scatratimint_->discretization()->add_dof_set(dofset) != ++number_dofsets)
      FOUR_C_THROW("Scalar transport discretization exhibits invalid number of dofsets!");
    scatratimint_->set_number_of_dof_set_growth(number_dofsets);
    // initialize linear solver for monolithic scatra-scatra interface coupling involving interface
    // layer growth
    if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
    {
      const int extendedsolver = Global::Problem::instance()
                                     ->scalar_transport_dynamic_params()
                                     .sublist("S2I COUPLING")
                                     .get<int>("INTLAYERGROWTH_LINEAR_SOLVER");
      if (extendedsolver < 1)
      {
        FOUR_C_THROW(
            "Invalid ID of linear solver for monolithic scatra-scatra interface coupling involving "
            "interface layer growth!");
      }
      extendedsolver_ = std::make_shared<Core::LinAlg::Solver>(
          Global::Problem::instance()->solver_params(extendedsolver),
          scatratimint_->discretization()->get_comm(),
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"));
    }
  }  // initialize scatra-scatra interface layer growth

  // safety check
  else if (intlayergrowth_evaluation_ != Inpar::S2I::growth_evaluation_none)
  {
    FOUR_C_THROW(
        "Cannot evaluate scatra-scatra interface coupling involving interface layer growth without "
        "specifying a corresponding boundary condition!");
  }

  // safety checks associated with adaptive time stepping for scatra-scatra interface layer growth
  if (intlayergrowth_timestep_ > 0.)
  {
    if (not scatratimint_->scatra_parameter_list()->get<bool>("ADAPTIVE_TIMESTEPPING"))
    {
      FOUR_C_THROW(
          "Adaptive time stepping for scatra-scatra interface layer growth requires "
          "ADAPTIVE_TIMESTEPPING flag to be set!");
    }
    if (!scatratimint_->discretization()->has_condition("S2IKineticsGrowth"))
    {
      FOUR_C_THROW(
          "Adaptive time stepping for scatra-scatra interface layer growth requires corresponding "
          "boundary condition!");
    }
    if (intlayergrowth_timestep_ >= scatratimint_->dt())
    {
      FOUR_C_THROW(
          "Adaptive time stepping for scatra-scatra interface layer growth requires that the "
          "modified time step size is smaller than the original time step size!");
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::build_block_map_extractors()
{
  if (matrixtype_ == Core::LinAlg::MatrixType::block_condition or
      matrixtype_ == Core::LinAlg::MatrixType::block_condition_dof)
  {
    // initialize reduced interface map extractors associated with blocks of global system matrix
    const int nblocks = scatratimint_->dof_block_maps()->num_maps();
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> blockmaps_slave(nblocks);
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> blockmaps_master(nblocks);
    for (int iblock = 0; iblock < nblocks; ++iblock)
    {
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps(2);
      maps[0] = scatratimint_->dof_block_maps()->map(iblock);
      maps[1] = not imortarredistribution_
                    ? interfacemaps_->map(1)
                    : std::dynamic_pointer_cast<const Core::LinAlg::Map>(islavemap_);
      blockmaps_slave[iblock] = Core::LinAlg::MultiMapExtractor::intersect_maps(maps);
      maps[1] = not imortarredistribution_
                    ? interfacemaps_->map(2)
                    : std::dynamic_pointer_cast<const Core::LinAlg::Map>(imastermap_);
      blockmaps_master[iblock] = Core::LinAlg::MultiMapExtractor::intersect_maps(maps);
    }
    blockmaps_slave_ =
        std::make_shared<Core::LinAlg::MultiMapExtractor>(*interfacemaps_->map(1), blockmaps_slave);
    blockmaps_slave_->check_for_valid_map_extractor();
    blockmaps_master_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
        *interfacemaps_->map(2), blockmaps_master);
    blockmaps_master_->check_for_valid_map_extractor();
  }
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::equip_extended_solver_with_null_space_info() const
{
  // consider extended linear solver for scatra-scatra interface layer growth if applicable
  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
  {
    // loop over blocks of scalar transport system matrix
    for (int iblock = 0; iblock < scatratimint_->dof_block_maps()->num_maps(); ++iblock)
    {
      // store number of current block as string, starting from 1
      std::ostringstream iblockstr;
      iblockstr << iblock + 1;

      // equip smoother for current matrix block with previously computed null space
      extendedsolver_->params().sublist("Inverse" + iblockstr.str()) =
          scatratimint_->solver()->params().sublist("Inverse" + iblockstr.str());
    }
    // store number of matrix block associated with scatra-scatra interface layer growth as string
    std::stringstream iblockstr;
    iblockstr << scatratimint_->dof_block_maps()->num_maps() + 1;

    // equip smoother for extra matrix block with null space associated with all degrees of freedom
    // for scatra-scatra interface layer growth
    Teuchos::ParameterList& mllist = extendedsolver_->params().sublist("Inverse" + iblockstr.str());
    mllist.set("PDE equations", 1);

    const auto nullspace = std::make_shared<Core::LinAlg::MultiVector<double>>(
        *(scatratimint_->dof_row_map(2)), 1, true);
    nullspace->put_scalar(1.0);

    // build the coordinate object related to the growth dofs
    const auto growth_cond_node_row_map = Core::Conditions::condition_node_row_map(
        *scatratimint_->discretization(), "S2IKineticsGrowth");
    const std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
        extract_retained_node_coordinates(
            *scatratimint_->discretization(), *growth_cond_node_row_map);
    mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", coordinates);

    mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspace);
  }
}

/*------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::solve(const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& increment,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,
    const std::shared_ptr<Core::LinAlg::Vector<double>>& phinp, const int iteration,
    Core::LinAlg::SolverParams& solver_params) const
{
  switch (intlayergrowth_evaluation_)
  {
    // no or semi-implicit treatment of scatra-scatra interface layer growth
    case Inpar::S2I::growth_evaluation_none:
    case Inpar::S2I::growth_evaluation_semi_implicit:
    {
      switch (couplingtype_)
      {
        case Inpar::S2I::coupling_matching_nodes:
        case Inpar::S2I::coupling_mortar_standard:
        case Inpar::S2I::coupling_mortar_condensed_petrov:
        case Inpar::S2I::coupling_mortar_condensed_bubnov:
        case Inpar::S2I::coupling_nts_standard:
        {
          // equilibrate global system of equations if necessary
          equilibration_->equilibrate_system(
              systemmatrix, residual, scatratimint_->dof_block_maps());

          // solve global system of equations
          solver_params.refactor = true;
          solver_params.reset = iteration == 1;
          solver->solve(systemmatrix, increment, residual, solver_params);

          // unequilibrate global increment vector if necessary
          equilibration_->unequilibrate_increment(increment);

          break;
        }

        case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
        case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
        {
          // check scalar transport system matrix
          std::shared_ptr<Core::LinAlg::SparseMatrix> sparsematrix =
              std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(systemmatrix);
          if (sparsematrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

          // assemble extended system matrix including rows and columns associated with Lagrange
          // multipliers
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>
              extendedsystemmatrix(*extendedmaps_, *extendedmaps_);
          extendedsystemmatrix.assign(0, 0, Core::LinAlg::DataAccess::Share, *sparsematrix);
          if (lmside_ == Inpar::S2I::side_slave)
          {
            Core::LinAlg::matrix_add(*D_, true, 1., extendedsystemmatrix.matrix(0, 1), 0.);
            Core::LinAlg::matrix_add(*M_, true, -1., extendedsystemmatrix.matrix(0, 1), 1.);
            Core::LinAlg::matrix_add(
                *Core::LinAlg::matrix_row_transform_gids(*islavematrix_, *extendedmaps_->map(1)),
                false, 1., extendedsystemmatrix.matrix(1, 0), 0.);
          }
          else
          {
            Core::LinAlg::matrix_add(*M_, true, -1., extendedsystemmatrix.matrix(0, 1), 0.);
            Core::LinAlg::matrix_add(*D_, true, 1., extendedsystemmatrix.matrix(0, 1), 1.);
            Core::LinAlg::matrix_add(
                *Core::LinAlg::matrix_row_transform_gids(*imastermatrix_, *extendedmaps_->map(1)),
                false, 1., extendedsystemmatrix.matrix(1, 0), 0.);
          }
          Core::LinAlg::matrix_add(*E_, true, -1., extendedsystemmatrix.matrix(1, 1), 0.);
          extendedsystemmatrix.complete();
          extendedsystemmatrix.matrix(0, 1).apply_dirichlet(
              *scatratimint_->dirich_maps()->cond_map(), false);

          std::shared_ptr<Core::LinAlg::Vector<double>> extendedresidual =
              std::make_shared<Core::LinAlg::Vector<double>>(*extendedmaps_->full_map());
          extendedmaps_->insert_vector(*scatratimint_->residual(), 0, *extendedresidual);
          extendedmaps_->insert_vector(*lmresidual_, 1, *extendedresidual);

          std::shared_ptr<Core::LinAlg::Vector<double>> extendedincrement =
              std::make_shared<Core::LinAlg::Vector<double>>(*extendedmaps_->full_map());
          extendedmaps_->insert_vector(*scatratimint_->increment(), 0, *extendedincrement);
          extendedmaps_->insert_vector(*lmincrement_, 1, *extendedincrement);

          // solve extended system of equations
          solver_params.refactor = true;
          solver_params.reset = iteration == 1;
          solver->solve(Core::Utils::shared_ptr_from_ref(extendedsystemmatrix), extendedincrement,
              extendedresidual, solver_params);

          // store solution
          extendedmaps_->extract_vector(*extendedincrement, 0, *increment);
          extendedmaps_->extract_vector(*extendedincrement, 1, *lmincrement_);

          // update Lagrange multipliers
          lm_->update(1., *lmincrement_, 1.);

          break;
        }

        default:
        {
          FOUR_C_THROW("Type of scatra-scatra interface coupling not recognized!");
        }
      }

      break;
    }

    // monolithic treatment of scatra-scatra interface layer growth
    case Inpar::S2I::growth_evaluation_monolithic:
    {
      switch (couplingtype_)
      {
        case Inpar::S2I::coupling_matching_nodes:
        {
          switch (matrixtype_)
          {
            case Core::LinAlg::MatrixType::sparse:
            {
              // check scalar transport system matrix
              const std::shared_ptr<const Core::LinAlg::SparseMatrix> sparsematrix =
                  std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(systemmatrix);
              if (sparsematrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

              // assemble extended system matrix including rows and columns associated with
              // scatra-scatra interface layer thickness variables
              extendedsystemmatrix_->assign(0, 0, Core::LinAlg::DataAccess::Share, *sparsematrix);
              extendedsystemmatrix_->assign(0, 1, Core::LinAlg::DataAccess::Share,
                  *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(scatragrowthblock_));
              extendedsystemmatrix_->assign(1, 0, Core::LinAlg::DataAccess::Share,
                  *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(growthscatrablock_));
              extendedsystemmatrix_->assign(
                  1, 1, Core::LinAlg::DataAccess::Share, *growthgrowthblock_);

              break;
            }

            case Core::LinAlg::MatrixType::block_condition:
            case Core::LinAlg::MatrixType::block_condition_dof:
            {
              // check scalar transport system matrix
              const std::shared_ptr<const Core::LinAlg::BlockSparseMatrixBase> blocksparsematrix =
                  std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(systemmatrix);
              if (blocksparsematrix == nullptr)
                FOUR_C_THROW("System matrix is not a block sparse matrix!");

              // extract number of matrix row or column blocks associated with scalar transport
              // field
              const int nblockmaps = static_cast<int>(scatratimint_->dof_block_maps()->num_maps());

              // construct extended system matrix by assigning matrix blocks
              for (int iblock = 0; iblock < nblockmaps; ++iblock)
              {
                for (int jblock = 0; jblock < nblockmaps; ++jblock)
                  extendedsystemmatrix_->assign(iblock, jblock, Core::LinAlg::DataAccess::Share,
                      blocksparsematrix->matrix(iblock, jblock));
                extendedsystemmatrix_->assign(iblock, nblockmaps, Core::LinAlg::DataAccess::Share,
                    std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                        scatragrowthblock_)
                        ->matrix(iblock, 0));
                extendedsystemmatrix_->assign(nblockmaps, iblock, Core::LinAlg::DataAccess::Share,
                    std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                        growthscatrablock_)
                        ->matrix(0, iblock));
              }
              extendedsystemmatrix_->assign(
                  nblockmaps, nblockmaps, Core::LinAlg::DataAccess::Share, *growthgrowthblock_);

              break;
            }

            default:
            {
              FOUR_C_THROW(
                  "Type of global system matrix for scatra-scatra interface coupling involving "
                  "interface layer growth not recognized!");
            }
          }

          // finalize extended system matrix
          extendedsystemmatrix_->complete();

          // assemble extended residual vector
          std::shared_ptr<Core::LinAlg::Vector<double>> extendedresidual =
              std::make_shared<Core::LinAlg::Vector<double>>(*extendedmaps_->full_map(), true);
          extendedmaps_->insert_vector(*scatratimint_->residual(), 0, *extendedresidual);
          extendedmaps_->insert_vector(*growthresidual_, 1, *extendedresidual);

          // perform finite-difference check if desired
          if (scatratimint_->fd_check_type() == Inpar::ScaTra::fdcheck_global_extended)
            fd_check(*extendedsystemmatrix_, *extendedresidual);

          // assemble extended increment vector
          std::shared_ptr<Core::LinAlg::Vector<double>> extendedincrement =
              std::make_shared<Core::LinAlg::Vector<double>>(*extendedmaps_->full_map(), true);
          extendedmaps_->insert_vector(*scatratimint_->increment(), 0, *extendedincrement);
          extendedmaps_->insert_vector(*growthincrement_, 1, *extendedincrement);

          // equilibrate global system of equations if necessary
          equilibration_->equilibrate_system(
              extendedsystemmatrix_, extendedresidual, extendedblockmaps_);

          // solve extended system of equations
          solver_params.refactor = true;
          solver_params.reset = iteration == 1;
          extendedsolver_->solve(
              extendedsystemmatrix_, extendedincrement, extendedresidual, solver_params);

          // unequilibrate global increment vector if necessary
          equilibration_->unequilibrate_increment(extendedincrement);

          // store solution
          extendedmaps_->extract_vector(*extendedincrement, 0, *increment);
          extendedmaps_->extract_vector(*extendedincrement, 1, *growthincrement_);

          // update state vector of discrete scatra-scatra interface layer thicknesses
          growthnp_->update(1., *growthincrement_, 1.);

          break;
        }

        default:
        {
          FOUR_C_THROW("Type of scatra-scatra interface coupling not recognized!");
          break;
        }
      }  // switch(couplingtype_)

      break;
    }

    default:
    {
      FOUR_C_THROW(
          "Unknown evaluation method for scatra-scatra interface coupling involving interface "
          "layer growth!");
      break;
    }
  }
}  // ScaTra::meshtying_strategy_s2_i::Solve

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::LinAlg::Solver& ScaTra::MeshtyingStrategyS2I::solver() const
{
  const Core::LinAlg::Solver* solver(nullptr);

  if (intlayergrowth_evaluation_ == Inpar::S2I::growth_evaluation_monolithic)
  {
    if (extendedsolver_ == nullptr) FOUR_C_THROW("Invalid linear solver!");
    solver = extendedsolver_.get();
  }

  else
  {
    if (scatratimint_->solver() == nullptr) FOUR_C_THROW("Invalid linear solver!");
    solver = scatratimint_->solver().get();
  }

  return *solver;
}  // ScaTra::meshtying_strategy_s2_i::Solver()

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MeshtyingStrategyS2I::fd_check(
    const Core::LinAlg::BlockSparseMatrixBase& extendedsystemmatrix,
    Core::LinAlg::Vector<double>& extendedresidual) const
{
  // initial screen output
  if (Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()) == 0)
  {
    std::cout << std::endl
              << "FINITE DIFFERENCE CHECK FOR EXTENDED SYSTEM MATRIX INVOLVING SCATRA-SCATRA "
                 "INTERFACE LAYER GROWTH"
              << std::endl;
  }

  // extract perturbation magnitude and relative tolerance
  const double fdcheckeps(scatratimint_->fd_check_eps());
  const double fdchecktol(scatratimint_->fd_check_tol());

  // create global state vector
  Core::LinAlg::Vector<double> statenp(*extendedmaps_->full_map(), true);
  extendedmaps_->insert_vector(*scatratimint_->phinp(), 0, statenp);
  extendedmaps_->insert_vector(*growthnp_, 1, statenp);

  // make a copy of global state vector to undo perturbations later
  Core::LinAlg::Vector<double> statenp_original(statenp);

  Core::LinAlg::SparseMatrix sysmat_original =
      Core::LinAlg::SparseMatrix(*extendedsystemmatrix.merge());
  sysmat_original.complete();

  // make a copy of system right-hand side vector
  Core::LinAlg::Vector<double> rhs_original(extendedresidual);

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  // loop over all columns of system matrix
  for (int colgid = 0; colgid <= sysmat_original.col_map().max_all_gid(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original.col_map().lid(colgid));
    int maxcollid(-1);
    maxcollid = Core::Communication::max_all(collid, scatratimint_->discretization()->get_comm());
    if (maxcollid < 0) continue;

    // fill global state vector with original state variables
    statenp.update(1., statenp_original, 0.);

    // impose perturbation
    if (statenp.get_map().my_gid(colgid)) statenp.sum_into_global_value(colgid, fdcheckeps);
    scatratimint_->phinp()->update(1., *extendedmaps_->extract_vector(statenp, 0), 0.);
    growthnp_->update(1., *extendedmaps_->extract_vector(statenp, 1), 0.);

    // calculate global right-hand side contributions based on perturbed state
    scatratimint_->assemble_mat_and_rhs();

    // assemble global residual vector
    extendedmaps_->insert_vector(*scatratimint_->residual(), 0, extendedresidual);
    extendedmaps_->insert_vector(*growthresidual_, 1, extendedresidual);

    // Now we compare the difference between the current entries in the system matrix
    // and their finite difference approximations according to
    // entries ?= (residual_perturbed - residual_original) / epsilon

    // Note that the residual_ vector actually denotes the right-hand side of the linear
    // system of equations, i.e., the negative system residual.
    // To account for errors due to numerical cancellation, we additionally consider
    // entries + residual_original / epsilon ?= residual_perturbed / epsilon

    // Note that we still need to evaluate the first comparison as well. For small entries in the
    // system matrix, the second comparison might yield good agreement in spite of the entries being
    // wrong!
    for (int rowlid = 0; rowlid < extendedmaps_->full_map()->num_my_elements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original.row_map().gid(rowlid);
      if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original.num_my_entries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original.extract_my_row_copy(
          rowlid, length, numentries, values.data(), indices.data());
      for (int ientry = 0; ientry < length; ++ientry)
      {
        if (sysmat_original.col_map().gid(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval = -extendedresidual.local_values_as_span()[rowlid] / fdcheckeps +
                           rhs_original.local_values_as_span()[rowlid] / fdcheckeps;

      // absolute and relative errors in first comparison
      const double abserr1 = entry - fdval;
      if (abs(abserr1) > maxabserr) maxabserr = abs(abserr1);
      double relerr1(0.);
      if (abs(entry) > 1.e-17)
        relerr1 = abserr1 / abs(entry);
      else if (abs(fdval) > 1.e-17)
        relerr1 = abserr1 / abs(fdval);
      if (abs(relerr1) > maxrelerr) maxrelerr = abs(relerr1);

      // evaluate first comparison
      if (abs(relerr1) > fdchecktol)
      {
        std::cout << std::setprecision(6);
        std::cout << "sysmat[" << rowgid << "," << colgid << "]:  " << entry << "   ";
        std::cout << "finite difference suggestion:  " << fdval << "   ";
        std::cout << "absolute error:  " << abserr1 << "   ";
        std::cout << "relative error:  " << relerr1 << std::endl;

        counter++;
      }

      // first comparison OK
      else
      {
        // left-hand side in second comparison
        const double left = entry - rhs_original.local_values_as_span()[rowlid] / fdcheckeps;

        // right-hand side in second comparison
        const double right = -extendedresidual.local_values_as_span()[rowlid] / fdcheckeps;

        // absolute and relative errors in second comparison
        const double abserr2 = left - right;
        if (abs(abserr2) > maxabserr) maxabserr = abs(abserr2);
        double relerr2(0.);
        if (abs(left) > 1.e-17)
          relerr2 = abserr2 / abs(left);
        else if (abs(right) > 1.e-17)
          relerr2 = abserr2 / abs(right);
        if (abs(relerr2) > maxrelerr) maxrelerr = abs(relerr2);

        // evaluate second comparison
        if (abs(relerr2) > fdchecktol)
        {
          std::cout << std::setprecision(6);
          std::cout << "sysmat[" << rowgid << "," << colgid << "]-rhs[" << rowgid
                    << "]/eps:  " << left << "   ";
          std::cout << "-rhs_perturbed[" << rowgid << "]/eps:  " << right << "   ";
          std::cout << "absolute error:  " << abserr2 << "   ";
          std::cout << "relative error:  " << relerr2 << std::endl;

          counter++;
        }
      }
    }
  }

  // communicate tracking variables
  int counterglobal(0);
  counterglobal =
      Core::Communication::sum_all(counter, scatratimint_->discretization()->get_comm());
  double maxabserrglobal(0.);
  maxabserrglobal =
      Core::Communication::max_all(maxabserr, scatratimint_->discretization()->get_comm());
  double maxrelerrglobal(0.);
  maxrelerrglobal =
      Core::Communication::max_all(maxrelerr, scatratimint_->discretization()->get_comm());

  // final screen output
  if (Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()) == 0)
  {
    if (counterglobal)
    {
      printf(
          "--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n", counterglobal);
      FOUR_C_THROW(
          "Finite difference check failed for extended system matrix involving scatra-scatra "
          "interface layer growth!");
    }
    else
    {
      printf(
          "--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",
          maxabserrglobal, maxrelerrglobal);
    }
  }

  // undo perturbations of state variables
  scatratimint_->phinp()->update(1., *extendedmaps_->extract_vector(statenp_original, 0), 0.);
  growthnp_->update(1., *extendedmaps_->extract_vector(statenp_original, 1), 0.);

  // recompute system matrix and right-hand side vector based on original state variables
  scatratimint_->assemble_mat_and_rhs();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::MortarCellInterface::MortarCellInterface(const Inpar::S2I::CouplingType& couplingtype,
    const Inpar::S2I::InterfaceSides& lmside, const int& numdofpernode_slave,
    const int& numdofpernode_master)
    : lmside_(lmside),
      couplingtype_(couplingtype),
      numdofpernode_slave_(numdofpernode_slave),
      numdofpernode_master_(numdofpernode_master)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalc<distype_s, distype_m>*
ScaTra::MortarCellCalc<distype_s, distype_m>::instance(const Inpar::S2I::CouplingType& couplingtype,
    const Inpar::S2I::InterfaceSides& lmside, const int& numdofpernode_slave,
    const int& numdofpernode_master, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
          const int& numdofpernode_slave, const int& numdofpernode_master)
      {
        return std::unique_ptr<MortarCellCalc<distype_s, distype_m>>(
            new MortarCellCalc<distype_s, distype_m>(
                couplingtype, lmside, numdofpernode_slave, numdofpernode_master));
      });

  return singleton_map[disname].instance(Core::Utils::SingletonAction::create, couplingtype, lmside,
      numdofpernode_slave, numdofpernode_master);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate(
    const Core::FE::Discretization& idiscret, Mortar::IntCell& cell, Mortar::Element& slaveelement,
    Mortar::Element& masterelement, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& cellmatrix1, Core::LinAlg::SerialDenseMatrix& cellmatrix2,
    Core::LinAlg::SerialDenseMatrix& cellmatrix3, Core::LinAlg::SerialDenseMatrix& cellmatrix4,
    Core::LinAlg::SerialDenseVector& cellvector1, Core::LinAlg::SerialDenseVector& cellvector2)
{
  // extract and evaluate action
  switch (Teuchos::getIntegralValue<Inpar::S2I::EvaluationActions>(params, "action"))
  {
    case Inpar::S2I::evaluate_mortar_matrices:
    {
      // evaluate mortar matrices
      evaluate_mortar_matrices(
          cell, slaveelement, masterelement, cellmatrix1, cellmatrix2, cellmatrix3);

      break;
    }

    case Inpar::S2I::evaluate_condition:
    {
      // evaluate and assemble interface linearizations and residuals
      evaluate_condition(idiscret, cell, slaveelement, masterelement, la_slave, la_master, params,
          cellmatrix1, cellmatrix2, cellmatrix3, cellmatrix4, cellvector1, cellvector2);

      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown action for mortar cell evaluation!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_nts(
    const Core::FE::Discretization& idiscret, const Mortar::Node& slavenode,
    const double& lumpedarea, Mortar::Element& slaveelement, Mortar::Element& masterelement,
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master,
    const Teuchos::ParameterList& params, Core::LinAlg::SerialDenseMatrix& ntsmatrix1,
    Core::LinAlg::SerialDenseMatrix& ntsmatrix2, Core::LinAlg::SerialDenseMatrix& ntsmatrix3,
    Core::LinAlg::SerialDenseMatrix& ntsmatrix4, Core::LinAlg::SerialDenseVector& ntsvector1,
    Core::LinAlg::SerialDenseVector& ntsvector2)
{
  // extract and evaluate action
  switch (Teuchos::getIntegralValue<Inpar::S2I::EvaluationActions>(params, "action"))
  {
    case Inpar::S2I::evaluate_condition_nts:
    {
      // extract condition from parameter list
      const Core::Conditions::Condition* condition =
          params.get<const Core::Conditions::Condition*>("condition");
      if (condition == nullptr)
        FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

      // extract nodal state variables associated with slave and master elements
      extract_node_values(idiscret, la_slave, la_master);

      // evaluate and assemble interface linearizations and residuals
      evaluate_condition_nts(*condition, slavenode, lumpedarea, slaveelement, masterelement,
          ephinp_slave_, ephinp_master_, ntsmatrix1, ntsmatrix2, ntsmatrix3, ntsmatrix4, ntsvector1,
          ntsvector2);

      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown action for evaluation of node-to-segment coupling!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_mortar_element(
    const Core::FE::Discretization& idiscret, Mortar::Element& element,
    Core::Elements::LocationArray& la, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseMatrix& elematrix3, Core::LinAlg::SerialDenseMatrix& elematrix4,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2)
{
  // extract and evaluate action
  switch (Teuchos::getIntegralValue<Inpar::S2I::EvaluationActions>(params, "action"))
  {
    case Inpar::S2I::evaluate_nodal_area_fractions:
    {
      // evaluate and assemble lumped interface area fractions associated with element nodes
      evaluate_nodal_area_fractions(element, elevector1);

      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown action for evaluation of mortar element!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
ScaTra::MortarCellCalc<distype_s, distype_m>::MortarCellCalc(
    const Inpar::S2I::CouplingType& couplingtype, const Inpar::S2I::InterfaceSides& lmside,
    const int& numdofpernode_slave, const int& numdofpernode_master)
    : MortarCellInterface(couplingtype, lmside, numdofpernode_slave, numdofpernode_master),
      scatraparamsboundary_(Discret::Elements::ScaTraEleParameterBoundary::instance("scatra")),
      ephinp_slave_(numdofpernode_slave,
          Core::LinAlg::Matrix<nen_slave_, 1>(Core::LinAlg::Initialization::zero)),
      ephinp_master_(numdofpernode_master,
          Core::LinAlg::Matrix<nen_master_, 1>(Core::LinAlg::Initialization::zero)),
      funct_slave_(Core::LinAlg::Initialization::zero),
      funct_master_(Core::LinAlg::Initialization::zero),
      shape_lm_slave_(Core::LinAlg::Initialization::zero),
      shape_lm_master_(Core::LinAlg::Initialization::zero),
      test_lm_slave_(Core::LinAlg::Initialization::zero),
      test_lm_master_(Core::LinAlg::Initialization::zero)
{
  // safety check
  if (nsd_slave_ != 2 or nsd_master_ != 2)
  {
    FOUR_C_THROW(
        "Scatra-scatra interface coupling with non-matching interface discretization currently "
        "only implemented for two-dimensional interface manifolds!");
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::extract_node_values(
    const Core::FE::Discretization& idiscret, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master)
{
  // extract nodal state variables associated with mortar integration cell
  extract_node_values(ephinp_slave_, ephinp_master_, idiscret, la_slave, la_master);
}


/*--------------------------------------------------------------------------*
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::extract_node_values(
    Core::LinAlg::Matrix<nen_slave_, 1>& estate_slave, const Core::FE::Discretization& idiscret,
    Core::Elements::LocationArray& la_slave, const std::string& statename, const int& nds) const
{
  // extract interface state vector from interface discretization
  const std::shared_ptr<const Core::LinAlg::Vector<double>> state =
      idiscret.get_state(nds, statename);
  if (state == nullptr)
    FOUR_C_THROW("Cannot extract state vector \"{}\" from interface discretization!", statename);

  // extract nodal state variables associated with slave element
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_slave_, 1>>(
      *state, estate_slave, la_slave[nds].lm_);
}


/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::extract_node_values(
    std::vector<Core::LinAlg::Matrix<nen_slave_, 1>>& estate_slave,
    std::vector<Core::LinAlg::Matrix<nen_master_, 1>>& estate_master,
    const Core::FE::Discretization& idiscret, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const std::string& statename, const int& nds) const
{
  // extract interface state vector from interface discretization
  const std::shared_ptr<const Core::LinAlg::Vector<double>> state =
      idiscret.get_state(nds, statename);
  if (state == nullptr)
    FOUR_C_THROW("Cannot extract state vector \"{}\" from interface discretization!", statename);

  // extract nodal state variables associated with slave and master elements
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_slave_, 1>>(
      *state, estate_slave, la_slave[nds].lm_);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_master_, 1>>(
      *state, estate_master, la_master[nds].lm_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double ScaTra::MortarCellCalc<distype_s, distype_m>::eval_shape_func_and_dom_int_fac_at_int_point(
    Mortar::Element& slaveelement, Mortar::Element& masterelement, Mortar::IntCell& cell,
    const Core::FE::IntPointsAndWeights<nsd_slave_>& intpoints, const int iquad)
{
  // reference coordinates of integration point
  std::array<double, nsd_slave_> coordinates_ref;
  for (int idim = 0; idim < nsd_slave_; ++idim)
    coordinates_ref[idim] = intpoints.ip().qxg[iquad][idim];

  // global coordinates of integration point
  std::array<double, nsd_slave_ + 1> coordinates_global;
  cell.local_to_global(coordinates_ref.data(), coordinates_global.data(), 0);

  // project integration point onto slave and master elements
  std::array<double, nsd_slave_> coordinates_slave;
  std::array<double, nsd_master_> coordinates_master;
  double dummy(0.);
  Mortar::Projector::impl(slaveelement)
      ->project_gauss_point_auxn_3d(
          coordinates_global.data(), cell.auxn(), slaveelement, coordinates_slave.data(), dummy);
  Mortar::Projector::impl(masterelement)
      ->project_gauss_point_auxn_3d(
          coordinates_global.data(), cell.auxn(), masterelement, coordinates_master.data(), dummy);

  // evaluate shape functions at current integration point on slave and master elements
  Coupling::VolMortar::Utils::shape_function<distype_s>(funct_slave_, coordinates_slave.data());
  Coupling::VolMortar::Utils::shape_function<distype_m>(funct_master_, coordinates_master.data());
  switch (couplingtype_)
  {
    case Inpar::S2I::coupling_mortar_standard:
    {
      // there actually aren't any Lagrange multipliers, but we still need to set pseudo Lagrange
      // multiplier test functions equal to the standard shape and test functions for correct
      // evaluation of the scatra-scatra interface coupling conditions
      test_lm_slave_ = funct_slave_;
      test_lm_master_ = funct_master_;

      break;
    }

    case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
    case Inpar::S2I::coupling_mortar_condensed_petrov:
    {
      // dual Lagrange multiplier shape functions combined with standard Lagrange multiplier test
      // functions
      if (lmside_ == Inpar::S2I::side_slave)
      {
        Coupling::VolMortar::Utils::dual_shape_function<distype_s>(
            shape_lm_slave_, coordinates_slave.data(), slaveelement);
        test_lm_slave_ = funct_slave_;
      }
      else
      {
        Coupling::VolMortar::Utils::dual_shape_function<distype_m>(
            shape_lm_master_, coordinates_master.data(), masterelement);
        test_lm_master_ = funct_master_;
      }

      break;
    }

    case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
    case Inpar::S2I::coupling_mortar_condensed_bubnov:
    {
      // dual Lagrange multiplier shape functions combined with dual Lagrange multiplier test
      // functions
      if (lmside_ == Inpar::S2I::side_slave)
      {
        Coupling::VolMortar::Utils::dual_shape_function<distype_s>(
            shape_lm_slave_, coordinates_slave.data(), slaveelement);
        test_lm_slave_ = shape_lm_slave_;
      }
      else
      {
        Coupling::VolMortar::Utils::dual_shape_function<distype_m>(
            shape_lm_master_, coordinates_master.data(), masterelement);
        test_lm_master_ = shape_lm_master_;
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Not yet implemented!");
      break;
    }
  }

  // integration weight
  const double weight = intpoints.ip().qwgt[iquad];

  // Jacobian determinant
  const double jacobian = cell.jacobian();

  // domain integration factor
  return jacobian * weight;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double ScaTra::MortarCellCalc<distype_s, distype_m>::eval_shape_func_and_dom_int_fac_at_int_point(
    Mortar::Element& element, const Core::FE::IntPointsAndWeights<nsd_slave_>& intpoints,
    const int iquad)
{
  // extract global coordinates of element nodes
  Core::LinAlg::Matrix<nsd_slave_ + 1, nen_slave_> coordinates_nodes;
  Core::Geo::fill_initial_position_array<distype_s, nsd_slave_ + 1,
      Core::LinAlg::Matrix<nsd_slave_ + 1, nen_slave_>>(&element, coordinates_nodes);

  // extract reference coordinates of integration point
  Core::LinAlg::Matrix<nsd_slave_, 1> coordinates_ref(intpoints.ip().qxg[iquad]);

  // evaluate slave-side shape functions and their first derivatives at integration point
  Core::LinAlg::Matrix<nsd_slave_, nen_slave_> deriv_slave;
  Core::FE::shape_function<distype_s>(coordinates_ref, funct_slave_);
  Core::FE::shape_function_deriv1<distype_s>(coordinates_ref, deriv_slave);

  // evaluate transposed Jacobian matrix at integration point
  Core::LinAlg::Matrix<nsd_slave_, nsd_slave_ + 1> jacobian;
  jacobian.multiply_nt(deriv_slave, coordinates_nodes);

  // evaluate metric tensor at integration point
  Core::LinAlg::Matrix<nsd_slave_, nsd_slave_> metrictensor;
  metrictensor.multiply_nt(jacobian, jacobian);

  // return domain integration factor, i.e., Jacobian determinant times integration weight, at
  // integration point
  return sqrt(metrictensor.determinant()) * intpoints.ip().qwgt[iquad];
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::eval_shape_func_at_slave_node(
    const Mortar::Node& slavenode, Mortar::Element& slaveelement, Mortar::Element& masterelement)
{
  // safety check
  if (couplingtype_ != Inpar::S2I::coupling_nts_standard)
    FOUR_C_THROW("This function should only be called when evaluating node-to-segment coupling!");

  // extract global ID of slave-side node
  const int& nodeid = slavenode.id();

  // find out index of slave-side node w.r.t. slave-side element
  int index(-1);
  for (int inode = 0; inode < slaveelement.num_node(); ++inode)
  {
    if (nodeid == slaveelement.nodes()[inode]->id())
    {
      index = inode;
      break;
    }
  }
  if (index == -1)
    FOUR_C_THROW("Couldn't find out index of slave-side node w.r.t. slave-side element!");

  // set slave-side shape function array according to node position
  funct_slave_.clear();
  funct_slave_(index) = 1.;

  // project slave-side node onto master-side element
  std::array<double, 2> coordinates_master;
  double dummy(0.);
  Mortar::Projector::impl(masterelement)
      ->project_gauss_point_auxn_3d(slavenode.x().data(), slavenode.mo_data().n(), masterelement,
          coordinates_master.data(), dummy);

  // evaluate master-side shape functions at projected node on master-side element
  Coupling::VolMortar::Utils::shape_function<distype_m>(funct_master_, coordinates_master.data());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_mortar_matrices(Mortar::IntCell& cell,
    Mortar::Element& slaveelement, Mortar::Element& masterelement,
    Core::LinAlg::SerialDenseMatrix& D, Core::LinAlg::SerialDenseMatrix& M,
    Core::LinAlg::SerialDenseMatrix& E)
{
  // safety check
  if (numdofpernode_slave_ != numdofpernode_master_)
    FOUR_C_THROW("Must have same number of degrees of freedom per node on slave and master sides!");

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // loop over all integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, iquad);

    if (lmside_ == Inpar::S2I::side_slave)
    {
      // loop over all degrees of freedom per node
      for (int k = 0; k < numdofpernode_slave_; ++k)
      {
        for (int vi = 0; vi < nen_slave_; ++vi)
        {
          const int row_slave = vi * numdofpernode_slave_ + k;

          switch (couplingtype_)
          {
            case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
            case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
            case Inpar::S2I::coupling_mortar_condensed_petrov:
            case Inpar::S2I::coupling_mortar_condensed_bubnov:
            {
              D(row_slave, row_slave) += shape_lm_slave_(vi) * fac;

              if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov or
                  couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
              {
                for (int ui = 0; ui < nen_slave_; ++ui)
                  E(row_slave, ui * numdofpernode_slave_ + k) +=
                      shape_lm_slave_(vi) * test_lm_slave_(ui) * fac;
              }

              break;
            }

            default:
            {
              for (int ui = 0; ui < nen_slave_; ++ui)
                D(row_slave, ui * numdofpernode_slave_ + k) +=
                    shape_lm_slave_(vi) * funct_slave_(ui) * fac;

              break;
            }
          }

          for (int ui = 0; ui < nen_master_; ++ui)
            M(row_slave, ui * numdofpernode_master_ + k) +=
                shape_lm_slave_(vi) * funct_master_(ui) * fac;
        }
      }
    }

    else
    {
      // loop over all degrees of freedom per node
      for (int k = 0; k < numdofpernode_master_; ++k)
      {
        for (int vi = 0; vi < nen_master_; ++vi)
        {
          const int row_master = vi * numdofpernode_master_ + k;

          switch (couplingtype_)
          {
            case Inpar::S2I::coupling_mortar_saddlepoint_petrov:
            case Inpar::S2I::coupling_mortar_saddlepoint_bubnov:
            case Inpar::S2I::coupling_mortar_condensed_petrov:
            case Inpar::S2I::coupling_mortar_condensed_bubnov:
            {
              D(row_master, row_master) += shape_lm_master_(vi) * fac;

              if (couplingtype_ == Inpar::S2I::coupling_mortar_saddlepoint_bubnov or
                  couplingtype_ == Inpar::S2I::coupling_mortar_condensed_bubnov)
              {
                for (int ui = 0; ui < nen_master_; ++ui)
                  E(row_master, ui * numdofpernode_master_ + k) +=
                      shape_lm_master_(vi) * test_lm_master_(ui) * fac;
              }

              break;
            }

            default:
            {
              for (int ui = 0; ui < nen_master_; ++ui)
                D(row_master, ui * numdofpernode_master_ + k) +=
                    shape_lm_master_(vi) * funct_master_(ui) * fac;

              break;
            }
          }

          for (int ui = 0; ui < nen_slave_; ++ui)
            M(row_master, ui * numdofpernode_slave_ + k) +=
                shape_lm_master_(vi) * funct_slave_(ui) * fac;
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_condition(
    const Core::FE::Discretization& idiscret, Mortar::IntCell& cell, Mortar::Element& slaveelement,
    Mortar::Element& masterelement, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
    Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
    Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m)
{
  // extract nodal state variables associated with slave and master elements
  extract_node_values(idiscret, la_slave, la_master);

  // safety check
  if (numdofpernode_slave_ != 1 or numdofpernode_master_ != 1)
  {
    FOUR_C_THROW(
        "Invalid number of degrees of freedom per node! Code should theoretically work for more "
        "than one degree of freedom per node, but not yet tested!");
  }

  // always in contact
  const double pseudo_contact_fac = 1.0;

  // determine quadrature rule
  const Core::FE::IntPointsAndWeights<2> intpoints(Core::FE::GaussRule2D::tri_7point);

  // loop over all integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = eval_shape_func_and_dom_int_fac_at_int_point(
        slaveelement, masterelement, cell, intpoints, iquad);

    // overall integration factors
    const double timefacfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac() * fac;
    const double timefacrhsfac =
        Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac_rhs() * fac;
    if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    Discret::Elements::ScaTraEleBoundaryCalc<
        distype_s>::template evaluate_s2_i_coupling_at_integration_point<distype_m>(ephinp_slave_,
        ephinp_master_, pseudo_contact_fac, funct_slave_, funct_master_, test_lm_slave_,
        test_lm_master_, numdofpernode_slave_, scatraparamsboundary_, timefacfac, timefacrhsfac,
        k_ss, k_sm, k_ms, k_mm, r_s, r_m);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_condition_nts(
    const Core::Conditions::Condition& condition, const Mortar::Node& slavenode,
    const double& lumpedarea, Mortar::Element& slaveelement, Mortar::Element& masterelement,
    const std::vector<Core::LinAlg::Matrix<nen_slave_, 1>>& ephinp_slave,
    const std::vector<Core::LinAlg::Matrix<nen_master_, 1>>& ephinp_master,
    Core::LinAlg::SerialDenseMatrix& k_ss, Core::LinAlg::SerialDenseMatrix& k_sm,
    Core::LinAlg::SerialDenseMatrix& k_ms, Core::LinAlg::SerialDenseMatrix& k_mm,
    Core::LinAlg::SerialDenseVector& r_s, Core::LinAlg::SerialDenseVector& r_m)
{
  // safety check
  if (numdofpernode_slave_ != 1 or numdofpernode_master_ != 1)
  {
    FOUR_C_THROW(
        "Invalid number of degrees of freedom per node! Code should theoretically work for more "
        "than one degree of freedom per node, but not yet tested!");
  }

  // evaluate shape functions at position of slave-side node
  eval_shape_func_at_slave_node(slavenode, slaveelement, masterelement);

  // always in contact
  const double pseudo_contact_fac = 1.0;

  // overall integration factors
  const double timefacfac =
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac() * lumpedarea;
  const double timefacrhsfac =
      Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")->time_fac_rhs() * lumpedarea;
  if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

  Discret::Elements::ScaTraEleBoundaryCalc<
      distype_s>::template evaluate_s2_i_coupling_at_integration_point<distype_m>(ephinp_slave,
      ephinp_master, pseudo_contact_fac, funct_slave_, funct_master_, funct_slave_, funct_master_,
      numdofpernode_slave_, scatraparamsboundary_, timefacfac, timefacrhsfac, k_ss, k_sm, k_ms,
      k_mm, r_s, r_m);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
void ScaTra::MortarCellCalc<distype_s, distype_m>::evaluate_nodal_area_fractions(
    Mortar::Element& slaveelement, Core::LinAlg::SerialDenseVector& areafractions)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_slave_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype_s>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions and domain integration factor at current integration point
    const double fac = eval_shape_func_and_dom_int_fac_at_int_point(slaveelement, intpoints, iquad);

    // compute integrals of shape functions to obtain lumped interface area fractions associated
    // with element nodes
    for (int inode = 0; inode < nen_slave_; ++inode)
      areafractions[inode * numdofpernode_slave_] += funct_slave_(inode) * fac;
  }  // loop over integration points
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
ScaTra::MortarCellAssemblyStrategy::MortarCellAssemblyStrategy(
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix1,
    const Inpar::S2I::InterfaceSides matrix1_side_rows,
    const Inpar::S2I::InterfaceSides matrix1_side_cols,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix2,
    const Inpar::S2I::InterfaceSides matrix2_side_rows,
    const Inpar::S2I::InterfaceSides matrix2_side_cols,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix3,
    const Inpar::S2I::InterfaceSides matrix3_side_rows,
    const Inpar::S2I::InterfaceSides matrix3_side_cols,
    std::shared_ptr<Core::LinAlg::SparseOperator> systemmatrix4,
    const Inpar::S2I::InterfaceSides matrix4_side_rows,
    const Inpar::S2I::InterfaceSides matrix4_side_cols,
    std::shared_ptr<Core::LinAlg::MultiVector<double>> systemvector1,
    const Inpar::S2I::InterfaceSides vector1_side,
    std::shared_ptr<Core::LinAlg::FEVector<double>> systemvector2,
    const Inpar::S2I::InterfaceSides vector2_side, const int nds_rows, const int nds_cols)
    : matrix1_side_rows_(matrix1_side_rows),
      matrix1_side_cols_(matrix1_side_cols),
      matrix2_side_rows_(matrix2_side_rows),
      matrix2_side_cols_(matrix2_side_cols),
      matrix3_side_rows_(matrix3_side_rows),
      matrix3_side_cols_(matrix3_side_cols),
      matrix4_side_rows_(matrix4_side_rows),
      matrix4_side_cols_(matrix4_side_cols),
      systemmatrix1_(std::move(systemmatrix1)),
      systemmatrix2_(std::move(systemmatrix2)),
      systemmatrix3_(std::move(systemmatrix3)),
      systemmatrix4_(std::move(systemmatrix4)),
      systemvector1_(std::move(systemvector1)),
      systemvector2_(std::move(systemvector2)),
      vector1_side_(vector1_side),
      vector2_side_(vector2_side),
      nds_rows_(nds_rows),
      nds_cols_(nds_cols)
{
}


/*----------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::assemble_cell_matrices_and_vectors(
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master,
    const int assembler_pid_master) const
{
  // assemble cell matrix 1 into system matrix 1
  if (assemble_matrix1())
    assemble_cell_matrix(systemmatrix1_, cellmatrix1_, matrix1_side_rows_, matrix1_side_cols_,
        la_slave, la_master, assembler_pid_master);

  // assemble cell matrix 2 into system matrix 2
  if (assemble_matrix2())
    assemble_cell_matrix(systemmatrix2_, cellmatrix2_, matrix2_side_rows_, matrix2_side_cols_,
        la_slave, la_master, assembler_pid_master);

  // assemble cell matrix 3 into system matrix 3
  if (assemble_matrix3())
    assemble_cell_matrix(systemmatrix3_, cellmatrix3_, matrix3_side_rows_, matrix3_side_cols_,
        la_slave, la_master, assembler_pid_master);

  // assemble cell matrix 4 into system matrix 4
  if (assemble_matrix4())
    assemble_cell_matrix(systemmatrix4_, cellmatrix4_, matrix4_side_rows_, matrix4_side_cols_,
        la_slave, la_master, assembler_pid_master);

  // assemble cell vector 1 into system vector 1
  if (assemble_vector1())
    assemble_cell_vector(
        *systemvector1_, cellvector1_, vector1_side_, la_slave, la_master, assembler_pid_master);

  // assemble cell vector 2 into system vector 2
  if (assemble_vector2())
    assemble_cell_vector(
        systemvector2_, cellvector2_, vector2_side_, la_slave, la_master, assembler_pid_master);
}


/*----------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::assemble_cell_matrix(
    const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
    const Core::LinAlg::SerialDenseMatrix& cellmatrix, const Inpar::S2I::InterfaceSides side_rows,
    const Inpar::S2I::InterfaceSides side_cols, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master, const int assembler_pid_master) const
{
  // determine location array associated with matrix columns
  Core::Elements::LocationArray& la_cols =
      side_cols == Inpar::S2I::side_slave ? la_slave : la_master;

  // assemble cell matrix into system matrix
  switch (side_rows)
  {
    case Inpar::S2I::side_slave:
    {
      systemmatrix->assemble(-1, la_cols[nds_cols_].stride_, cellmatrix, la_slave[nds_rows_].lm_,
          la_slave[nds_rows_].lmowner_, la_cols[nds_cols_].lm_);
      break;
    }

    case Inpar::S2I::side_master:
    {
      std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(systemmatrix)
          ->fe_assemble(cellmatrix, la_master[nds_rows_].lm_,
              std::vector<int>(la_master[nds_rows_].lmowner_.size(), assembler_pid_master),
              la_cols[nds_cols_].lm_);
      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid interface side!");
      break;
    }
  }
}


/*----------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::assemble_cell_vector(
    Core::LinAlg::MultiVector<double>& systemvector,
    const Core::LinAlg::SerialDenseVector& cellvector, const Inpar::S2I::InterfaceSides side,
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master,
    const int assembler_pid_master) const
{
  // assemble cell vector into system vector
  switch (side)
  {
    case Inpar::S2I::side_slave:
    {
      if (systemvector.num_vectors() != 1)
        FOUR_C_THROW("Invalid number of vectors inside Core::LinAlg::MultiVector<double>!");
      Core::LinAlg::assemble((systemvector).get_vector(nds_rows_), cellvector,
          la_slave[nds_rows_].lm_, la_slave[nds_rows_].lmowner_);

      break;
    }

    case Inpar::S2I::side_master:
    {
      if (assembler_pid_master == Core::Communication::my_mpi_rank(systemvector.get_comm()))
      {
        FOUR_C_ASSERT(false, "Don't know what to do! Need a FEVector.");
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid interface side!");
      break;
    }
  }
}


/*----------------------------------------------------------------------------------*
 *----------------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::assemble_cell_vector(
    const std::shared_ptr<Core::LinAlg::FEVector<double>>& systemvector,
    const Core::LinAlg::SerialDenseVector& cellvector, const Inpar::S2I::InterfaceSides side,
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master,
    const int assembler_pid_master) const
{
  // assemble cell vector into system vector
  switch (side)
  {
    case Inpar::S2I::side_slave:
    {
      FOUR_C_ASSERT(false, "Don't know what to do, does not work for FE-Vectors.");
      break;
    }

    case Inpar::S2I::side_master:
    {
      if (assembler_pid_master == Core::Communication::my_mpi_rank(systemvector->get_comm()))
      {
        std::dynamic_pointer_cast<Core::LinAlg::FEVector<double>>(systemvector)
            ->sum_into_global_values(static_cast<int>(la_master[nds_rows_].lm_.size()),
                la_master[nds_rows_].lm_.data(), cellvector.values());
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid interface side!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::init_cell_matrices_and_vectors(
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master)
{
  // initialize system matrix 1
  if (assemble_matrix1())
    init_cell_matrix(cellmatrix1_, matrix1_side_rows_, matrix1_side_cols_, la_slave, la_master);

  // initialize system matrix 2
  if (assemble_matrix2())
    init_cell_matrix(cellmatrix2_, matrix2_side_rows_, matrix2_side_cols_, la_slave, la_master);

  // initialize system matrix 3
  if (assemble_matrix3())
    init_cell_matrix(cellmatrix3_, matrix3_side_rows_, matrix3_side_cols_, la_slave, la_master);

  // initialize system matrix 4
  if (assemble_matrix4())
    init_cell_matrix(cellmatrix4_, matrix4_side_rows_, matrix4_side_cols_, la_slave, la_master);

  // initialize system vector 1
  if (assemble_vector1()) init_cell_vector(cellvector1_, vector1_side_, la_slave, la_master);

  // initialize system vector 2
  if (assemble_vector2()) init_cell_vector(cellvector2_, vector2_side_, la_slave, la_master);
}


/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::init_cell_matrix(
    Core::LinAlg::SerialDenseMatrix& cellmatrix, const Inpar::S2I::InterfaceSides side_rows,
    const Inpar::S2I::InterfaceSides side_cols, Core::Elements::LocationArray& la_slave,
    Core::Elements::LocationArray& la_master) const
{
  // determine number of matrix rows and number of matrix columns
  const int nrows = side_rows == Inpar::S2I::side_slave ? la_slave[nds_rows_].size()
                                                        : la_master[nds_rows_].size();
  const int ncols = side_cols == Inpar::S2I::side_slave ? la_slave[nds_cols_].size()
                                                        : la_master[nds_cols_].size();

  // reshape cell matrix if necessary
  if (cellmatrix.numRows() != nrows or cellmatrix.numCols() != ncols)
  {
    cellmatrix.shape(nrows, ncols);
  }

  // simply zero out otherwise
  else
    cellmatrix.putScalar(0.0);
}


/*---------------------------------------------------------------------------*
 *---------------------------------------------------------------------------*/
void ScaTra::MortarCellAssemblyStrategy::init_cell_vector(
    Core::LinAlg::SerialDenseVector& cellvector, const Inpar::S2I::InterfaceSides side,
    Core::Elements::LocationArray& la_slave, Core::Elements::LocationArray& la_master) const
{
  // determine number of vector components
  const int ndofs =
      side == Inpar::S2I::side_slave ? la_slave[nds_rows_].size() : la_master[nds_rows_].size();

  // reshape cell vector if necessary
  if (cellvector.length() != ndofs)
  {
    cellvector.size(ndofs);

    // simply zero out otherwise
  }
  else
    cellvector.putScalar(0.0);
}


// forward declarations
template class ScaTra::MortarCellCalc<Core::FE::CellType::tri3, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalc<Core::FE::CellType::tri3, Core::FE::CellType::quad4>;
template class ScaTra::MortarCellCalc<Core::FE::CellType::quad4, Core::FE::CellType::tri3>;
template class ScaTra::MortarCellCalc<Core::FE::CellType::quad4, Core::FE::CellType::quad4>;

FOUR_C_NAMESPACE_CLOSE
