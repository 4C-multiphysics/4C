// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_sti_monolithic.hpp"

#include "4C_constraint_framework_embeddedmesh_solid_to_solid_mortar_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_converter.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_assemblestrategy.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_equilibrate.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_method_parameters.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"
#include "4C_sti_input.hpp"
#include "4C_sti_monolithic_evaluate_OffDiag.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
STI::Monolithic::Monolithic(MPI_Comm comm, const Teuchos::ParameterList& stidyn,
    const Teuchos::ParameterList& scatradyn, const Teuchos::ParameterList& solverparams,
    const Teuchos::ParameterList& solverparams_scatra,
    const Teuchos::ParameterList& solverparams_thermo)
    : Algorithm(comm, stidyn, scatradyn, solverparams_scatra, solverparams_thermo),
      restol_(fieldparameters_->sublist("NONLINEAR").get<double>("ABSTOLRES")),
      maps_(nullptr),
      condensationthermo_(stidyn.get<bool>("THERMO_CONDENSATION")),
      systemmatrix_(nullptr),
      matrixtype_(Teuchos::getIntegralValue<Core::LinAlg::MatrixType>(
          stidyn.sublist("MONOLITHIC"), "MATRIXTYPE")),
      scatrathermoblockdomain_(nullptr),
      scatrathermoblockinterface_(nullptr),
      thermoscatrablockdomain_(nullptr),
      thermoscatrablockinterface_(nullptr),
      blockmaps_(nullptr),
      blockmapthermo_(nullptr),
      increment_(nullptr),
      residual_(nullptr),
      dtele_(0.),
      dtsolve_(0.),
      solver_(std::make_shared<Core::LinAlg::Solver>(solverparams, comm,
          Global::Problem::instance()->solver_params_callback(),
          Teuchos::getIntegralValue<Core::IO::Verbositylevel>(
              Global::Problem::instance()->io_params(), "VERBOSITY"))),
      invrowsums_(nullptr),
      icoupscatra_(nullptr),
      icoupthermo_(nullptr),
      islavetomasterrowtransformscatraod_(nullptr),
      islavetomastercoltransformthermood_(nullptr),
      islavetomasterrowtransformthermood_(nullptr),
      equilibration_(nullptr)
{
  // safety checks
  if (!scatra_field()->is_incremental())
    FOUR_C_THROW("Must have incremental solution approach for scatra-thermo interaction!");
  if (thermo_field()->system_matrix() == nullptr)
    FOUR_C_THROW("System matrix associated with temperature field must be a sparse matrix!");

  // set control parameters for Newton-Raphson iteration loop
  itermax_ = fieldparameters_->sublist("NONLINEAR").get<int>("ITEMAX");
  itertol_ = fieldparameters_->sublist("NONLINEAR").get<double>("CONVTOL");


  // extract coupling adapters for scatra-scatra interface mesh tying
  if (scatra_field()->s2i_meshtying() and
      strategyscatra_->coupling_type() == Inpar::S2I::coupling_matching_nodes)
  {
    icoupscatra_ = strategyscatra_->coupling_adapter();
    icoupthermo_ = strategythermo_->coupling_adapter();
  }

  // initialize map associated with single thermo block of global system matrix
  std::shared_ptr<const Core::LinAlg::Map> mapthermo(nullptr);
  if (condensationthermo_)
  {
    mapthermo = Core::LinAlg::merge_map(
        *strategythermo_->interface_maps()->map(0), *strategythermo_->interface_maps()->map(2));
  }
  else
  {
    mapthermo = thermo_field()->dof_row_map();
  }

  // initialize global map extractor
  maps_ = std::make_shared<Core::LinAlg::MapExtractor>(
      *Core::LinAlg::merge_map(*scatra_field()->discretization()->dof_row_map(), *mapthermo, false),
      mapthermo, scatra_field()->dof_row_map());

  // check global map extractor
  maps_->check_for_valid_map_extractor();

  // initialize global increment vector for Newton-Raphson iteration
  increment_ = Core::LinAlg::create_vector(*dof_row_map(), true);

  // initialize global residual vector
  residual_ = Core::LinAlg::create_vector(*dof_row_map(), true);

  // initialize transformation operators
  islavetomasterrowtransformscatraod_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();
  islavetomastercoltransformthermood_ = std::make_shared<Coupling::Adapter::MatrixColTransform>();
  islavetomasterrowtransformthermood_ = std::make_shared<Coupling::Adapter::MatrixRowTransform>();

  // merge slave and master side block maps for interface matrix for thermo and scatra
  std::shared_ptr<Core::LinAlg::Map> interface_map_scatra(nullptr);
  std::shared_ptr<Core::LinAlg::Map> interface_map_thermo(nullptr);
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapscatrainterface(nullptr);
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapthermointerface(nullptr);
  std::shared_ptr<Core::LinAlg::MultiMapExtractor> blockmapthermointerfaceslave(nullptr);

  if (scatra_field()->s2i_meshtying())
  {
    // merge slave and master side full maps for interface matrix for thermo and scatra
    interface_map_scatra = Core::LinAlg::MultiMapExtractor::merge_maps(
        {strategyscatra_->interface_maps()->map(1), strategyscatra_->interface_maps()->map(2)});
    interface_map_thermo = Core::LinAlg::MultiMapExtractor::merge_maps(
        {strategythermo_->interface_maps()->map(1), strategythermo_->interface_maps()->map(2)});
    // build block map for thermo interface by using full thermo map
    blockmapthermointerface =
        std::make_shared<Core::LinAlg::MultiMapExtractor>(*interface_map_thermo,
            std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, interface_map_thermo));
    blockmapthermointerface->check_for_valid_map_extractor();
    blockmapthermointerfaceslave = std::make_shared<Core::LinAlg::MultiMapExtractor>(
        *strategythermo_->interface_maps()->map(1),
        std::vector<std::shared_ptr<const Core::LinAlg::Map>>(
            1, strategythermo_->interface_maps()->map(1)));
    blockmapthermointerfaceslave->check_for_valid_map_extractor();
  }

  // initialize map extractors associated with blocks of global system matrix
  switch (scatra_field()->matrix_type())
  {
    // one single main-diagonal matrix block associated with scalar transport field
    case Core::LinAlg::MatrixType::sparse:
    {
      blockmaps_ = maps_;

      if (scatra_field()->s2i_meshtying())
      {
        blockmapscatrainterface =
            std::make_shared<Core::LinAlg::MultiMapExtractor>(*interface_map_scatra,
                std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, interface_map_scatra));
        blockmapscatrainterface->check_for_valid_map_extractor();
      }
      break;
    }

    // several main-diagonal matrix blocks associated with scalar transport field
    case Core::LinAlg::MatrixType::block_condition:
    {
      // extract maps underlying main-diagonal matrix blocks associated with scalar transport field
      const int nblockmapsscatra = static_cast<int>(scatra_field()->dof_block_maps()->num_maps());
      std::vector<std::shared_ptr<const Core::LinAlg::Map>> blockmaps(nblockmapsscatra + 1);
      for (int iblockmap = 0; iblockmap < nblockmapsscatra; ++iblockmap)
        blockmaps[iblockmap] = scatra_field()->dof_block_maps()->map(iblockmap);

      // extract map underlying single main-diagonal matrix block associated with temperature field
      blockmaps[nblockmapsscatra] = mapthermo;

      // initialize map extractor associated with blocks of global system matrix
      blockmaps_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(*dof_row_map(), blockmaps);

      // initialize map extractor associated with all degrees of freedom inside temperature field
      blockmapthermo_ = std::make_shared<Core::LinAlg::MultiMapExtractor>(
          *thermo_field()->discretization()->dof_row_map(),
          std::vector<std::shared_ptr<const Core::LinAlg::Map>>(1, thermo_field()->dof_row_map()));

      // safety check
      blockmapthermo_->check_for_valid_map_extractor();
      if (scatra_field()->s2i_meshtying())
      {
        // build block map for scatra interface by merging slave and master side for each block
        std::vector<std::shared_ptr<const Core::LinAlg::Map>> partial_blockmapscatrainterface(
            nblockmapsscatra, nullptr);
        for (int iblockmap = 0; iblockmap < nblockmapsscatra; ++iblockmap)
        {
          partial_blockmapscatrainterface.at(iblockmap) =
              Core::LinAlg::MultiMapExtractor::merge_maps(
                  {strategyscatra_->block_maps_slave().map(iblockmap),
                      strategyscatra_->block_maps_master().map(iblockmap)});
        }
        blockmapscatrainterface = std::make_shared<Core::LinAlg::MultiMapExtractor>(
            *interface_map_scatra, partial_blockmapscatrainterface);
        blockmapscatrainterface->check_for_valid_map_extractor();
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // safety check
  blockmaps_->check_for_valid_map_extractor();

  // perform initializations associated with global system matrix
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      // safety check
      const bool allowed_block_system_solvers = solver().params().isSublist("Teko Parameters") or
                                                solver().params().isSublist("MueLu Parameters");
      FOUR_C_ASSERT_ALWAYS(allowed_block_system_solvers,
          "Global system matrix with block structure requires MueLu or Teko block preconditioner!");

      // initialize global system matrix
      systemmatrix_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *blockmaps_, *blockmaps_, 81, false, true);

      // feed block preconditioner with null space information for each block of global block system
      // matrix
      build_null_spaces();

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      // safety check
      if (scatra_field()->matrix_type() != Core::LinAlg::MatrixType::sparse)
        FOUR_C_THROW("Incompatible matrix type associated with scalar transport field!");

      // initialize global system matrix
      systemmatrix_ = std::make_shared<Core::LinAlg::SparseMatrix>(*dof_row_map(), 27, false, true);

      // feed AMG preconditioner with null space information associated with global system matrix if
      // applicable
      compute_null_space_if_necessary(solver_->params());

      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scatra-thermo interaction not recognized!");
      break;
    }
  }

  // initialize scatra-thermo block and thermo-scatra block of global system matrix
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      // initialize scatra-thermo blocks
      scatrathermoblockdomain_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *blockmapthermo_, *scatra_field()->dof_block_maps(), 81, false, true);
      scatrathermoblockinterface_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *blockmapthermo_, *blockmapscatrainterface, 81, false, true);

      // initialize thermo-scatra blocks
      thermoscatrablockdomain_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatra_field()->dof_block_maps(), *blockmapthermo_, 81, false, true);
      thermoscatrablockinterface_ = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatra_field()->dof_block_maps(), *blockmapthermointerface, 81, false, true);

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      // initialize scatra-thermo blocks
      scatrathermoblockdomain_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *scatra_field()->discretization()->dof_row_map(), 27, false, true);
      scatrathermoblockinterface_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(*interface_map_scatra, 27, false, true);

      // initialize thermo-scatra blocks
      thermoscatrablockdomain_ = std::make_shared<Core::LinAlg::SparseMatrix>(
          *thermo_field()->discretization()->dof_row_map(), 27, false, true);
      thermoscatrablockinterface_ =
          std::make_shared<Core::LinAlg::SparseMatrix>(*interface_map_thermo, 27, false, true);

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
    }
  }

  // STI algorithm has non deforming mesh
  static constexpr bool isAle = false;

  // initialize OD evaluation strategy
  scatrathermooffdiagcoupling_ = STI::build_scatra_thermo_off_diag_coupling(
      strategyscatra_->coupling_type(), blockmapthermo_, blockmapthermointerface,
      blockmapthermointerfaceslave, maps_->map(0), maps_->map(1), interface_map_scatra,
      interface_map_thermo, isAle, strategyscatra_, strategythermo_, scatra_, thermo_);

  // instantiate appropriate equilibration class
  auto equilibration_method =
      std::vector<Core::LinAlg::EquilibrationMethod>(1, scatra_field()->equilibration_method());
  equilibration_ =
      Core::LinAlg::build_equilibration(matrixtype_, equilibration_method, maps_->full_map());
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::fd_check()
{
  // initial screen output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
    std::cout << std::endl << "FINITE DIFFERENCE CHECK FOR STI SYSTEM MATRIX" << std::endl;

  // create global state vector
  std::shared_ptr<Core::LinAlg::Vector<double>> statenp(
      Core::LinAlg::create_vector(*dof_row_map(), true));
  maps_->insert_vector(*scatra_field()->phinp(), 0, *statenp);
  maps_->insert_vector(*thermo_field()->phinp(), 1, *statenp);

  // make a copy of global state vector to undo perturbations later
  auto statenp_original = std::make_shared<Core::LinAlg::Vector<double>>(*statenp);

  std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_original = nullptr;
  if (std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(systemmatrix_) != nullptr)
  {
    sysmat_original = std::make_shared<Core::LinAlg::SparseMatrix>(
        *(std::static_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(systemmatrix_)->merge()));
  }
  else
    FOUR_C_THROW("Global system matrix must be a block sparse matrix!");
  sysmat_original->complete();

  // make a copy of system right-hand side vector
  Core::LinAlg::Vector<double> rhs_original(*residual_);

  // initialize counter for system matrix entries with failing finite difference check
  int counter(0);

  // initialize tracking variable for maximum absolute and relative errors
  double maxabserr(0.);
  double maxrelerr(0.);

  for (int colgid = 0; colgid <= sysmat_original->col_map().max_all_gid(); ++colgid)
  {
    // check whether current column index is a valid global column index and continue loop if not
    int collid(sysmat_original->col_map().lid(colgid));
    int maxcollid(-1);
    maxcollid = Core::Communication::max_all(collid, get_comm());
    if (maxcollid < 0) continue;

    // fill global state vector with original state variables
    statenp->update(1., *statenp_original, 0.);

    // impose perturbation
    if (statenp->get_map().my_gid(colgid))
      statenp->sum_into_global_value(colgid, scatra_field()->fd_check_eps());
    scatra_field()->phinp()->update(1., *maps_->extract_vector(*statenp, 0), 0.);
    thermo_field()->phinp()->update(1., *maps_->extract_vector(*statenp, 1), 0.);

    // carry perturbation over to state vectors at intermediate time stages if necessary
    scatra_field()->compute_intermediate_values();
    thermo_field()->compute_intermediate_values();

    // calculate element right-hand side vector for perturbed state
    assemble_mat_and_rhs();

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
    for (int rowlid = 0; rowlid < dof_row_map()->num_my_elements(); ++rowlid)
    {
      // get global index of current matrix row
      const int rowgid = sysmat_original->row_map().gid(rowlid);
      if (rowgid < 0) FOUR_C_THROW("Invalid global ID of matrix row!");

      // get relevant entry in current row of original system matrix
      double entry(0.);
      int length = sysmat_original->num_my_entries(rowlid);
      int numentries;
      std::vector<double> values(length);
      std::vector<int> indices(length);
      sysmat_original->extract_my_row_copy(
          rowlid, length, numentries, values.data(), indices.data());
      for (int ientry = 0; ientry < length; ++ientry)
      {
        if (sysmat_original->col_map().gid(indices[ientry]) == colgid)
        {
          entry = values[ientry];
          break;
        }
      }

      // finite difference suggestion (first divide by epsilon and then add for better conditioning)
      const double fdval =
          -residual_->local_values_as_span()[rowlid] / scatra_field()->fd_check_eps() +
          rhs_original.local_values_as_span()[rowlid] / scatra_field()->fd_check_eps();

      // confirm accuracy of first comparison
      if (abs(fdval) > 1.e-17 and abs(fdval) < 1.e-15)
        FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

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
      if (abs(relerr1) > scatra_field()->fd_check_tol())
      {
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
        const double left =
            entry - rhs_original.local_values_as_span()[rowlid] / scatra_field()->fd_check_eps();

        // right-hand side in second comparison
        const double right =
            -residual_->local_values_as_span()[rowlid] / scatra_field()->fd_check_eps();

        // confirm accuracy of second comparison
        if (abs(right) > 1.e-17 and abs(right) < 1.e-15)
          FOUR_C_THROW("Finite difference check involves values too close to numerical zero!");

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
        if (abs(relerr2) > scatra_field()->fd_check_tol())
        {
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
  counterglobal = Core::Communication::sum_all(counter, get_comm());
  double maxabserrglobal(0.);
  maxabserrglobal = Core::Communication::max_all(maxabserr, get_comm());
  double maxrelerrglobal(0.);
  maxrelerrglobal = Core::Communication::max_all(maxrelerr, get_comm());

  // final screen output
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    if (counterglobal)
    {
      printf(
          "--> FAILED AS LISTED ABOVE WITH %d CRITICAL MATRIX ENTRIES IN TOTAL\n\n", counterglobal);
      FOUR_C_THROW("Finite difference check failed for STI system matrix!");
    }
    else
    {
      printf(
          "--> PASSED WITH MAXIMUM ABSOLUTE ERROR %+12.5e AND MAXIMUM RELATIVE ERROR %+12.5e\n\n",
          maxabserrglobal, maxrelerrglobal);
    }
  }

  // undo perturbations of state variables
  scatra_field()->phinp()->update(1., *maps_->extract_vector(*statenp_original, 0), 0.);
  scatra_field()->compute_intermediate_values();
  thermo_field()->phinp()->update(1., *maps_->extract_vector(*statenp_original, 1), 0.);
  thermo_field()->compute_intermediate_values();

  // recompute system matrix and right-hand side vector based on original state variables
  assemble_mat_and_rhs();
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::output_matrix_to_file(
    const std::shared_ptr<const Core::LinAlg::SparseOperator> sparseoperator, const int precision,
    const double tolerance)
{
  // safety check
  if (!sparseoperator->filled()) FOUR_C_THROW("Sparse operator must be filled for output!");

  // extract communicator
  MPI_Comm comm = Core::Communication::unpack_epetra_comm(sparseoperator->Comm());

  // determine whether sparse matrix or block sparse matrix should be output
  const auto sparsematrix =
      std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(sparseoperator);
  const auto blocksparsematrix =
      std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(sparseoperator);
  if (sparsematrix == nullptr and blocksparsematrix == nullptr)
    FOUR_C_THROW("Unknown type of sparse operator!");

  // extract row map
  const Core::LinAlg::Map& rowmap =
      sparsematrix != nullptr ? sparsematrix->row_map() : blocksparsematrix->full_row_map();

  // safety check
  if (!rowmap.unique_gids()) FOUR_C_THROW("Row map of matrix must be non-overlapping!");

  // copy global IDs of matrix rows stored on current processor into vector
  std::vector<int> myrowgids(rowmap.num_my_elements(), 0);
  int* myglobalelements = rowmap.my_global_elements();
  std::copy(myglobalelements, myglobalelements + rowmap.num_my_elements(), myrowgids.data());

  // communicate global IDs
  std::vector<int> rowgids;
  Core::LinAlg::allreduce_vector(myrowgids, rowgids, comm);

  // retain communicated global IDs only on processor with ID 0
  if (Core::Communication::my_mpi_rank(comm)) rowgids.clear();

  // create full row map on processor with ID 0
  const Core::LinAlg::Map fullrowmap(
      -1, static_cast<int>(rowgids.size()), rowgids.size() ? rowgids.data() : nullptr, 0, comm);

  // import matrix to processor with ID 0
  Core::LinAlg::SparseMatrix crsmatrix(fullrowmap, 0);
  if (sparsematrix != nullptr)
  {
    crsmatrix.import(*sparsematrix, Core::LinAlg::Import(fullrowmap, rowmap), Insert);
  }
  else
  {
    for (int i = 0; i < blocksparsematrix->rows(); ++i)
    {
      for (int j = 0; j < blocksparsematrix->cols(); ++j)
      {
        crsmatrix.import(blocksparsematrix->matrix(i, j),
            Core::LinAlg::Import(fullrowmap, blocksparsematrix->range_map(i)), Insert);
      }
    }
  }

  // let processor with ID 0 output matrix to file
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    // set file name
    std::ostringstream nproc;
    nproc << Core::Communication::num_mpi_ranks(comm);
    const std::string filename(Global::Problem::instance()->output_control_file()->file_name() +
                               ".matrix_" + nproc.str() + "proc.csv");

    // open file and write header at beginning
    std::ofstream file;
    file.open(filename.c_str(), std::fstream::trunc);
    file << std::setprecision(precision) << std::scientific << "RowGIDs,ColumnGIDs,Values"
         << std::endl;

    // write matrix to file
    for (int rowlid = 0; rowlid < crsmatrix.num_my_rows(); ++rowlid)
    {
      // extract global ID of current matrix row
      const int rowgid = fullrowmap.gid(rowlid);

      // extract current matrix row
      int numentries;
      double* values;
      int* indices;
      crsmatrix.extract_global_row_view(rowgid, numentries, values, indices);

      // sort entries in current matrix row in ascending order of column global ID via map
      std::map<int, double> entries;

      // loop over all entries in current matrix row
      for (int j = 0; j < numentries; ++j)
        // add current matrix entry to map
        entries[indices[j]] = values[j];

      // loop over all sorted entries in current matrix row
      for (auto& entry : entries)
      {
        // write current matrix entry to file
        if (std::abs(entry.second) > tolerance)
          file << rowgid << "," << entry.first << "," << entry.second << std::endl;
      }
    }

    // close file
    file.close();
  }

  // wait until output is complete
  Core::Communication::barrier(comm);

  // throw error to abort simulation for debugging
  FOUR_C_THROW("Matrix was output to *.csv file!");
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::output_vector_to_file(
    const Core::LinAlg::MultiVector<double>& vector, const int precision, const double tolerance)
{
  // extract communicator
  MPI_Comm comm = vector.get_comm();

  // extract vector map
  const Core::LinAlg::Map& map = vector.get_map();

  // safety check
  if (!map.unique_gids())
    FOUR_C_THROW(
        "Vector output to *.csv file currently only works for non-overlapping vector maps!");

  // copy global IDs of vector components stored on current processor into vector
  std::vector<int> mygids(map.num_my_elements(), 0);
  int* myglobalelements = map.my_global_elements();
  std::copy(myglobalelements, myglobalelements + map.num_my_elements(), mygids.data());

  // communicate global IDs
  std::vector<int> gids;
  Core::LinAlg::allreduce_vector(mygids, gids, comm);

  // retain communicated global IDs only on processor with ID 0
  if (Core::Communication::my_mpi_rank(comm)) gids.clear();

  // create full vector map on processor with ID 0
  const Core::LinAlg::Map fullmap(
      -1, static_cast<int>(gids.size()), gids.size() ? gids.data() : nullptr, 0, comm);

  // export vector to processor with ID 0
  Core::LinAlg::MultiVector<double> fullvector(fullmap, vector.num_vectors(), true);
  Core::LinAlg::export_to(vector, fullvector);

  // let processor with ID 0 output vector to file
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    // set file name
    std::ostringstream nproc;
    nproc << Core::Communication::num_mpi_ranks(comm);
    const std::string filename(Global::Problem::instance()->output_control_file()->file_name() +
                               ".vector_" + nproc.str() + "proc.csv");

    // open file and write header at beginning
    std::ofstream file;
    file.open(filename.c_str(), std::fstream::trunc);
    file << std::setprecision(precision) << std::scientific << "GIDs,Values" << std::endl;

    // write vector to file
    for (int lid = 0; lid < fullvector.local_length(); ++lid)
    {
      // inner loop index
      int j(-1);

      // check output omission tolerance
      for (j = 0; j < fullvector.num_vectors(); ++j)
        if (std::abs(fullvector(j).local_values_as_span()[lid]) > tolerance) break;

      // perform output if applicable
      if (j < fullvector.num_vectors())
      {
        // write global ID of current vector component
        file << fullmap.gid(lid);

        // loop over all subvectors
        for (j = 0; j < fullvector.num_vectors(); ++j)
          // write current vector component to file
          file << "," << fullvector(j).local_values_as_span()[lid];

        // output line break
        file << std::endl;
      }
    }

    // close file
    file.close();
  }

  // wait until output is complete
  Core::Communication::barrier(comm);

  // throw error to abort simulation for debugging
  FOUR_C_THROW("Vector was output to *.csv file!");
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::assemble_mat_and_rhs()
{
  // pass thermo degrees of freedom to scatra discretization
  transfer_thermo_to_scatra(thermo_field()->phiafnp());

  // build system matrix and residual for scatra field
  scatra_field()->prepare_linear_solve();

  // pass scatra degrees of freedom to thermo discretization
  transfer_scatra_to_thermo(scatra_field()->phiafnp());

  // build system matrix and residual for thermo field
  thermo_field()->prepare_linear_solve();

  // evaluate scatra-thermo-OD coupling. Contributions from domain
  scatrathermooffdiagcoupling_->evaluate_off_diag_block_scatra_thermo_domain(
      scatrathermoblockdomain_);

  // evaluate scatra-thermo-OD coupling. Contributions from interface
  if (scatra_field()->s2i_meshtying())
    scatrathermooffdiagcoupling_->evaluate_off_diag_block_scatra_thermo_interface(
        scatrathermoblockinterface_);

  // evaluate thermo-scatra-OD coupling. Contributions from domain
  scatrathermooffdiagcoupling_->evaluate_off_diag_block_thermo_scatra_domain(
      thermoscatrablockdomain_);

  // evaluate thermo-scatra-OD coupling. Contributions from interface
  if (scatra_field()->s2i_meshtying())
    scatrathermooffdiagcoupling_->evaluate_off_diag_block_thermo_scatra_interface(
        thermoscatrablockinterface_);

  // OD blocks containing assembled domain and interface contributions
  std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermo_domain_interface(nullptr);
  std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatra_domain_interface(nullptr);

  // assemble interface and domain contributions of OD blocks
  assemble_domain_interface_off_diag(scatrathermo_domain_interface, thermoscatra_domain_interface);

  // apply Dirichlet BCs to OD blocks
  apply_dirichlet_off_diag(scatrathermo_domain_interface, thermoscatra_domain_interface);

  // build global system matrix
  switch (matrixtype_)
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      // check global system matrix
      std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> blocksystemmatrix =
          std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(systemmatrix_);
      if (blocksystemmatrix == nullptr) FOUR_C_THROW("System matrix is not a block matrix!");

      switch (scatra_field()->matrix_type())
      {
        case Core::LinAlg::MatrixType::block_condition:
        {
          // extract number of matrix row or column blocks associated with scalar transport field
          const int nblockmapsscatra = scatra_field()->dof_block_maps()->num_maps();

          // construct global system matrix by assigning matrix blocks
          for (int iblock = 0; iblock < nblockmapsscatra; ++iblock)
          {
            for (int jblock = 0; jblock < nblockmapsscatra; ++jblock)
              blocksystemmatrix->assign(iblock, jblock, Core::LinAlg::DataAccess::Share,
                  scatra_field()->block_system_matrix()->matrix(iblock, jblock));

            // perform second condensation before assigning matrix blocks
            if (condensationthermo_)
            {
              const auto& scatrathermoblock =
                  std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                      scatrathermo_domain_interface)
                      ->matrix(iblock, 0);
              Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                  scatrathermoblock.range_map(), *maps_->map(1), 1.0, nullptr, nullptr,
                  blocksystemmatrix->matrix(iblock, nblockmapsscatra));

              switch (strategyscatra_->coupling_type())
              {
                case Inpar::S2I::coupling_matching_nodes:
                {
                  Coupling::Adapter::CouplingSlaveConverter converter(*icoupthermo_);
                  Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                      scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1),
                      1.0, nullptr, &converter, blocksystemmatrix->matrix(iblock, nblockmapsscatra),
                      true, true);
                  break;
                }
                case Inpar::S2I::coupling_mortar_standard:
                {
                  // initialize temporary matrix for slave-side columns of current matrix block
                  Core::LinAlg::SparseMatrix scatrathermocolsslave(scatrathermoblock.row_map(), 81);

                  // fill temporary matrix for slave-side columns of current matrix block
                  Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                      scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1),
                      1.0, nullptr, nullptr, scatrathermocolsslave);

                  // finalize temporary matrix for slave-side columns of current matrix block
                  scatrathermocolsslave.complete(
                      *strategythermo_->interface_maps()->map(1), scatrathermoblock.row_map());

                  // transform and assemble temporary matrix for slave-side columns of current
                  // matrix block
                  blocksystemmatrix->matrix(iblock, nblockmapsscatra)
                      .add(*Core::LinAlg::matrix_multiply(
                               scatrathermocolsslave, false, *strategythermo_->p(), false, true),
                          false, 1.0, 1.0);

                  break;
                }
                default:
                {
                  FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
                  break;
                }
              }

              const auto& thermoscatrablock =
                  std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                      thermoscatra_domain_interface)
                      ->matrix(0, iblock);
              Coupling::Adapter::MatrixLogicalSplitAndTransform()(thermoscatrablock, *maps_->map(1),
                  thermoscatrablock.domain_map(), 1.0, nullptr, nullptr,
                  blocksystemmatrix->matrix(nblockmapsscatra, iblock));
            }

            // assign matrix blocks directly
            else
            {
              blocksystemmatrix->assign(iblock, nblockmapsscatra, Core::LinAlg::DataAccess::Share,
                  std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                      scatrathermo_domain_interface)
                      ->matrix(iblock, 0));

              blocksystemmatrix->assign(nblockmapsscatra, iblock, Core::LinAlg::DataAccess::Share,
                  std::dynamic_pointer_cast<const Core::LinAlg::BlockSparseMatrixBase>(
                      thermoscatra_domain_interface)
                      ->matrix(0, iblock));
            }
          }

          // perform second condensation before assigning thermo-thermo matrix block
          if (condensationthermo_)
          {
            Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermo_field()->system_matrix(),
                *maps_->map(1), *maps_->map(1), 1.0, nullptr, nullptr,
                blocksystemmatrix->matrix(nblockmapsscatra, nblockmapsscatra));

            switch (strategyscatra_->coupling_type())
            {
              case Inpar::S2I::coupling_matching_nodes:
              {
                Coupling::Adapter::CouplingSlaveConverter converter(*icoupthermo_);
                Coupling::Adapter::MatrixLogicalSplitAndTransform()(
                    *thermo_field()->system_matrix(), *maps_->map(1),
                    *strategythermo_->interface_maps()->map(1), 1.0, nullptr, &converter,
                    blocksystemmatrix->matrix(nblockmapsscatra, nblockmapsscatra), true, true);
                break;
              }
              case Inpar::S2I::coupling_mortar_standard:
              {
                // initialize temporary matrix for slave-side columns of thermo-thermo matrix block
                Core::LinAlg::SparseMatrix thermothermocolsslave(*maps_->map(1), 81);

                // fill temporary matrix for slave-side columns of thermo-thermo matrix block
                Coupling::Adapter::MatrixLogicalSplitAndTransform()(
                    *thermo_field()->system_matrix(), *maps_->map(1),
                    *strategythermo_->interface_maps()->map(1), 1.0, nullptr, nullptr,
                    thermothermocolsslave);

                // finalize temporary matrix for slave-side columns of thermo-thermo matrix block
                thermothermocolsslave.complete(
                    *strategythermo_->interface_maps()->map(1), *maps_->map(1));

                // transform and assemble temporary matrix for slave-side columns of thermo-thermo
                // matrix block
                blocksystemmatrix->matrix(nblockmapsscatra, nblockmapsscatra)
                    .add(*Core::LinAlg::matrix_multiply(
                             thermothermocolsslave, false, *strategythermo_->p(), false, true),
                        false, 1.0, 1.0);
                break;
              }
              default:
              {
                FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
                break;
              }
            }
          }

          // assign thermo-thermo matrix block directly
          else
          {
            blocksystemmatrix->assign(nblockmapsscatra, nblockmapsscatra,
                Core::LinAlg::DataAccess::Share, *thermo_field()->system_matrix());
          }

          break;
        }

        case Core::LinAlg::MatrixType::sparse:
        {
          // construct global system matrix by assigning matrix blocks
          blocksystemmatrix->assign(
              0, 0, Core::LinAlg::DataAccess::Share, *scatra_field()->system_matrix());

          // perform second condensation before assigning matrix blocks
          if (condensationthermo_)
          {
            const auto& scatrathermoblock =
                *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(
                    scatrathermo_domain_interface);
            Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                scatrathermoblock.range_map(), *maps_->map(1), 1.0, nullptr, nullptr,
                blocksystemmatrix->matrix(0, 1));

            Coupling::Adapter::MatrixLogicalSplitAndTransform()(
                *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(
                    thermoscatra_domain_interface),
                *maps_->map(1), Core::LinAlg::Map(thermoscatra_domain_interface->domain_map()), 1.0,
                nullptr, nullptr, blocksystemmatrix->matrix(1, 0));

            Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermo_field()->system_matrix(),
                *maps_->map(1), *maps_->map(1), 1.0, nullptr, nullptr,
                blocksystemmatrix->matrix(1, 1));

            switch (strategyscatra_->coupling_type())
            {
              case Inpar::S2I::coupling_matching_nodes:
              {
                Coupling::Adapter::CouplingSlaveConverter converter(*icoupthermo_);
                Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                    scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1), 1.0,
                    nullptr, &converter, blocksystemmatrix->matrix(0, 1), true, true);

                Coupling::Adapter::MatrixLogicalSplitAndTransform()(
                    *thermo_field()->system_matrix(), *maps_->map(1),
                    *strategythermo_->interface_maps()->map(1), 1.0, nullptr, &converter,
                    blocksystemmatrix->matrix(1, 1), true, true);

                break;
              }

              case Inpar::S2I::coupling_mortar_standard:
              {
                // initialize temporary matrix for slave-side columns of scatra-thermo matrix block
                Core::LinAlg::SparseMatrix scatrathermocolsslave(
                    *scatra_field()->discretization()->dof_row_map(), 81);

                // fill temporary matrix for slave-side columns of scatra-thermo matrix block
                Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                    scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1), 1.0,
                    nullptr, nullptr, scatrathermocolsslave);

                // finalize temporary matrix for slave-side columns of scatra-thermo matrix block
                scatrathermocolsslave.complete(*strategythermo_->interface_maps()->map(1),
                    *scatra_field()->discretization()->dof_row_map());

                // transform and assemble temporary matrix for slave-side columns of scatra-thermo
                // matrix block
                blocksystemmatrix->matrix(0, 1).add(
                    *Core::LinAlg::matrix_multiply(
                        scatrathermocolsslave, false, *strategythermo_->p(), false, true),
                    false, 1.0, 1.0);

                // initialize temporary matrix for slave-side columns of thermo-thermo matrix block
                Core::LinAlg::SparseMatrix thermothermocolsslave(*maps_->map(1), 81);

                // fill temporary matrix for slave-side columns of thermo-thermo matrix block
                Coupling::Adapter::MatrixLogicalSplitAndTransform()(
                    *thermo_field()->system_matrix(), *maps_->map(1),
                    *strategythermo_->interface_maps()->map(1), 1.0, nullptr, nullptr,
                    thermothermocolsslave);

                // finalize temporary matrix for slave-side columns of thermo-thermo matrix block
                thermothermocolsslave.complete(
                    *strategythermo_->interface_maps()->map(1), *maps_->map(1));

                // transform and assemble temporary matrix for slave-side columns of thermo-thermo
                // matrix block
                blocksystemmatrix->matrix(1, 1).add(
                    *Core::LinAlg::matrix_multiply(
                        thermothermocolsslave, false, *strategythermo_->p(), false, true),
                    false, 1.0, 1.0);
                break;
              }

              default:
              {
                FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
                break;
              }
            }
          }

          // assign matrix blocks directly
          else
          {
            blocksystemmatrix->assign(0, 1, Core::LinAlg::DataAccess::Share,
                *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
                    scatrathermo_domain_interface));
            blocksystemmatrix->assign(1, 0, Core::LinAlg::DataAccess::Share,
                *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
                    thermoscatra_domain_interface));
            blocksystemmatrix->assign(
                1, 1, Core::LinAlg::DataAccess::Share, *thermo_field()->system_matrix());
          }

          break;
        }

        default:
        {
          FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
          break;
        }
        case Core::LinAlg::MatrixType::undefined:
        case Core::LinAlg::MatrixType::block_field:
        case Core::LinAlg::MatrixType::block_condition_dof:
          break;
      }

      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      // check global system matrix
      auto systemmatrix = std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(systemmatrix_);
      if (systemmatrix == nullptr) FOUR_C_THROW("System matrix is not a sparse matrix!");

      // construct global system matrix by adding matrix blocks
      systemmatrix->add(*scatra_field()->system_matrix(), false, 1.0, 0.0);

      // perform second condensation before adding matrix blocks
      if (condensationthermo_)
      {
        const auto& scatrathermoblock =
            *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(
                scatrathermo_domain_interface);
        Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
            scatrathermoblock.range_map(), *maps_->map(1), 1.0, nullptr, nullptr, *systemmatrix,
            true, true);

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(
            *std::dynamic_pointer_cast<const Core::LinAlg::SparseMatrix>(
                thermoscatra_domain_interface),
            *maps_->map(1), Core::LinAlg::Map(thermoscatra_domain_interface->domain_map()), 1.0,
            nullptr, nullptr, *systemmatrix, true, true);

        Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermo_field()->system_matrix(),
            *maps_->map(1), *maps_->map(1), 1.0, nullptr, nullptr, *systemmatrix, true, true);

        switch (strategyscatra_->coupling_type())
        {
          case Inpar::S2I::coupling_matching_nodes:
          {
            Coupling::Adapter::CouplingSlaveConverter converter(*icoupthermo_);
            Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1), 1.0,
                nullptr, &converter, *systemmatrix, true, true);

            Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermo_field()->system_matrix(),
                *maps_->map(1), *strategythermo_->interface_maps()->map(1), 1.0, nullptr,
                &converter, *systemmatrix, true, true);

            break;
          }

          case Inpar::S2I::coupling_mortar_standard:
          {
            // initialize temporary matrix for slave-side columns of global system matrix
            Core::LinAlg::SparseMatrix systemmatrixcolsslave(*dof_row_map(), 81);

            // fill temporary matrix for slave-side columns of global system matrix
            Coupling::Adapter::MatrixLogicalSplitAndTransform()(scatrathermoblock,
                scatrathermoblock.range_map(), *strategythermo_->interface_maps()->map(1), 1.0,
                nullptr, nullptr, systemmatrixcolsslave);

            Coupling::Adapter::MatrixLogicalSplitAndTransform()(*thermo_field()->system_matrix(),
                *maps_->map(1), *strategythermo_->interface_maps()->map(1), 1.0, nullptr, nullptr,
                systemmatrixcolsslave, true, true);

            // finalize temporary matrix for slave-side columns of global system matrix
            systemmatrixcolsslave.complete(
                *strategythermo_->interface_maps()->map(1), *dof_row_map());

            // transform and assemble temporary matrix for slave-side columns of global system
            // matrix
            systemmatrix->add(*Core::LinAlg::matrix_multiply(
                                  systemmatrixcolsslave, false, *strategythermo_->p(), false, true),
                false, 1.0, 1.0);
            break;
          }

          default:
          {
            FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
            break;
          }
        }
      }

      // add matrix blocks directly
      else
      {
        systemmatrix->add(*scatrathermo_domain_interface, false, 1.0, 1.0);
        systemmatrix->add(*thermoscatra_domain_interface, false, 1.0, 1.0);
        systemmatrix->add(*thermo_field()->system_matrix(), false, 1.0, 1.0);
      }
      break;
    }

    default:
    {
      FOUR_C_THROW("Type of global system matrix for scatra-thermo interaction not recognized!");
      break;
    }
  }

  // finalize global system matrix
  systemmatrix_->complete();

  // create full monolithic right-hand side vector
  maps_->insert_vector(*scatra_field()->residual(), 0, *residual_);
  std::shared_ptr<Core::LinAlg::Vector<double>> thermoresidual(nullptr);
  if (condensationthermo_)
  {
    thermoresidual = std::make_shared<Core::LinAlg::Vector<double>>(*maps_->map(1));
    Core::LinAlg::export_to(*thermo_field()->residual(), *thermoresidual);
  }
  else
  {
    thermoresidual = thermo_field()->residual();
  }
  maps_->insert_vector(*thermoresidual, 1, *residual_);
}  // STI::Monolithic::assemble_mat_and_rhs()


/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void STI::Monolithic::build_null_spaces() const
{
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      Core::FE::compute_null_space_if_necessary(
          *scatra_field()->discretization(), solver_->params(), true);
      scatra_field()->build_block_null_spaces(*solver_, 0);
      break;
    }

    case Core::LinAlg::MatrixType::sparse:
    {
      Teuchos::ParameterList& blocksmootherparams = solver_->params().sublist("Inverse1");
      Core::LinearSolver::Parameters::compute_solver_parameters(
          *scatra_field()->discretization(), blocksmootherparams);

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid matrix type associated with scalar transport field!");
      break;
    }
  }

  // store number of matrix block associated with temperature field as string
  std::stringstream iblockstr;
  iblockstr << blockmaps_->num_maps();

  if (solver_->params().isSublist("MueLu Parameters"))
  {
    solver_->params()
        .sublist("Inverse" + iblockstr.str())
        .set("MueLu Parameters", solver_->params().sublist("MueLu Parameters"));
  }

  Teuchos::ParameterList& blocksmootherparams =
      solver_->params().isSublist("MueLu Parameters")
          ? solver_->params().sublist("Inverse" + iblockstr.str()).sublist("MueLu Parameters")
          : solver_->params().sublist("Inverse" + iblockstr.str());

  Core::LinearSolver::Parameters::compute_solver_parameters(
      *thermo_field()->discretization(), blocksmootherparams);

  // reduce full null space to match degrees of freedom associated with thermo matrix block if
  // necessary
  if (condensationthermo_)
    Core::LinearSolver::Parameters::fix_null_space("Block " + iblockstr.str(),
        *thermo_field()->discretization()->dof_row_map(), *maps_->map(1), blocksmootherparams);
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::compute_null_space_if_necessary(Teuchos::ParameterList& solverparams) const
{
  // compute point-based null space information for MueLu preconditioner
  if (solverparams.isSublist("MueLu Parameters"))
  {
    // extract and fill parameter list for MueLu preconditioner
    Teuchos::ParameterList& mllist = solverparams.sublist("MueLu Parameters", true);
    mllist.set("PDE equations", 1);

    std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
        std::make_shared<Core::LinAlg::MultiVector<double>>(dof_row_map().operator*(), 1, true);
    nullspace->put_scalar(1.0);

    mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullspace);

    std::shared_ptr<Core::LinAlg::MultiVector<double>> coordinates =
        extract_node_coordinates(*scatra_field()->discretization());

    mllist.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("Coordinates", coordinates);
  }
}  // STI::Monolithic::compute_null_space_if_necessary

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> STI::Monolithic::dof_row_map() const
{
  return maps_->full_map();
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
bool STI::Monolithic::exit_newton_raphson()
{
  bool is_converged(false);

  // perform Newton-Raphson convergence check depending on type of scalar transport
  switch (Teuchos::getIntegralValue<STI::ScaTraTimIntType>(*stiparameters_, "SCATRATIMINTTYPE"))
  {
    case STI::ScaTraTimIntType::elch:
    {
      // compute L2 norm of concentration state vector
      double concdofnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_other_vector(*scatra_field()->phinp())
          ->norm_2(&concdofnorm);

      // compute L2 norm of concentration residual vector
      double concresnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_other_vector(*maps_->extract_vector(*residual_, 0))
          ->norm_2(&concresnorm);

      // compute L2 norm of concentration increment vector
      double concincnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_other_vector(*maps_->extract_vector(*increment_, 0))
          ->norm_2(&concincnorm);

      // compute L2 norm of potential state vector
      double potdofnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_cond_vector(*scatra_field()->phinp())
          ->norm_2(&potdofnorm);

      // compute L2 norm of potential residual vector
      double potresnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_cond_vector(*maps_->extract_vector(*residual_, 0))
          ->norm_2(&potresnorm);

      // compute L2 norm of potential increment vector
      double potincnorm(0.);
      scatra_field()
          ->splitter()
          ->extract_cond_vector(*maps_->extract_vector(*increment_, 0))
          ->norm_2(&potincnorm);

      // compute L2 norm of thermo state vector
      double thermodofnorm(0.);
      thermo_field()->phinp()->norm_2(&thermodofnorm);

      // compute L2 norm of thermo residual vector
      double thermoresnorm(0.);
      maps_->extract_vector(*residual_, 1)->norm_2(&thermoresnorm);

      // compute L2 norm of thermo increment vector
      double thermoincnorm(0.);
      maps_->extract_vector(*increment_, 1)->norm_2(&thermoincnorm);

      // safety checks
      if (std::isnan(concdofnorm) or std::isnan(concresnorm) or std::isnan(concincnorm) or
          std::isnan(potdofnorm) or std::isnan(potresnorm) or std::isnan(potincnorm) or
          std::isnan(thermodofnorm) or std::isnan(thermoresnorm) or std::isnan(thermoincnorm))
        FOUR_C_THROW("Vector norm is not a number!");
      if (std::isinf(concdofnorm) or std::isinf(concresnorm) or std::isinf(concincnorm) or
          std::isinf(potdofnorm) or std::isinf(potresnorm) or std::isinf(potincnorm) or
          std::isinf(thermodofnorm) or std::isinf(thermoresnorm) or std::isinf(thermoincnorm))
        FOUR_C_THROW("Vector norm is infinity!");

      // prevent division by zero
      if (concdofnorm < 1.e-10) concdofnorm = 1.e-10;
      if (potdofnorm < 1.e-10) potdofnorm = 1.e-10;
      if (thermodofnorm < 1.e-10) thermodofnorm = 1.e-10;

      // first Newton-Raphson iteration
      if (iter_ == 1)
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          // print header of convergence table to screen
          std::cout << "+------------+-------------------+--------------+--------------+-----------"
                       "---+--------------+--------------+--------------+"
                    << std::endl;
          std::cout << "|- step/max -|- tolerance[norm] -|-- conc-res --|-- conc-inc --|-- pot-res "
                       "---|-- pot-inc ---|- thermo-res -|- thermo-inc -|"
                    << std::endl;

          // print first line of convergence table to screen
          // solution increment not yet available during first Newton-Raphson iteration
          std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
                    << std::setw(10) << std::setprecision(3) << std::scientific << itertol_
                    << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
                    << concresnorm << "   |      --      | " << std::setw(10)
                    << std::setprecision(3) << std::scientific << potresnorm
                    << "   |      --      | " << std::setw(10) << std::setprecision(3)
                    << std::scientific << thermoresnorm << "   |      --      | "
                    << "(       --      , te = " << std::setw(10) << std::setprecision(3) << dtele_
                    << ")" << std::endl;
        }
      }

      // subsequent Newton-Raphson iterations
      else
      {
        // print current line of convergence table to screen
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          std::cout << "|  " << std::setw(3) << iter_ << "/" << std::setw(3) << itermax_ << "   | "
                    << std::setw(10) << std::setprecision(3) << std::scientific << itertol_
                    << "[L_2 ]  | " << std::setw(10) << std::setprecision(3) << std::scientific
                    << concresnorm << "   | " << std::setw(10) << std::setprecision(3)
                    << std::scientific << concincnorm / concdofnorm << "   | " << std::setw(10)
                    << std::setprecision(3) << std::scientific << potresnorm << "   | "
                    << std::setw(10) << std::setprecision(3) << std::scientific
                    << potincnorm / potdofnorm << "   | " << std::setw(10) << std::setprecision(3)
                    << std::scientific << thermoresnorm << "   | " << std::setw(10)
                    << std::setprecision(3) << std::scientific << thermoincnorm / thermodofnorm
                    << "   | (ts = " << std::setw(10) << std::setprecision(3) << dtsolve_
                    << ", te = " << std::setw(10) << std::setprecision(3) << dtele_ << ")"
                    << std::endl;
        }

        // convergence check
        if (concresnorm <= itertol_ and potresnorm <= itertol_ and thermoresnorm <= itertol_ and
            concincnorm / concdofnorm <= itertol_ and potincnorm / potdofnorm <= itertol_ and
            thermoincnorm / thermodofnorm <= itertol_)
          // exit Newton-Raphson iteration upon convergence
          is_converged = true;
      }

      // exit Newton-Raphson iteration when residuals are small enough to prevent unnecessary
      // additional solver calls
      if (concresnorm < restol_ and potresnorm < restol_ and thermoresnorm < restol_)
        is_converged = true;

      // print warning to screen if maximum number of Newton-Raphson iterations is reached without
      // convergence
      if (iter_ == itermax_ and !is_converged)
      {
        if (Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          std::cout << "+------------+-------------------+--------------+--------------+-----------"
                       "---+--------------+--------------+--------------+"
                    << std::endl;
          std::cout << "|                     Newton-Raphson method has not converged after a "
                       "maximum number of "
                    << std::setw(2) << itermax_ << " iterations!                     |"
                    << std::endl;
        }

        // proceed to next time step
        is_converged = true;
      }

      // print finish line of convergence table to screen
      if (is_converged and Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        std::cout << "+------------+-------------------+--------------+--------------+-------------"
                     "-+--------------+--------------+--------------+"
                  << std::endl
                  << std::endl;
      }

      break;
    }

    default:
    {
      FOUR_C_THROW("Type of scalar transport not yet available!");
      break;
    }
  }

  return is_converged;
}  // STI::Monolithic::exit_newton_raphson()

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::prepare_time_step()
{
  // call base class routine
  Algorithm::prepare_time_step();

  // print time step information to screen
  scatra_field()->print_time_step_info();
}

/*--------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------*/
void STI::Monolithic::solve()
{
  // initialize counter for Newton-Raphson iterations
  iter_ = 0;

  // start Newton-Raphson iteration
  while (true)
  {
    // update iteration counter
    ++iter_;

    // store time before evaluating elements and assembling global system of equations
    double time = timer_->wallTime();

    // assemble global system of equations
    assemble_mat_and_rhs();

    // determine time needed for evaluating elements and assembling global system of equations,
    // and take maximum over all processors via communication
    double mydtele = timer_->wallTime() - time;
    dtele_ = Core::Communication::max_all(mydtele, get_comm());

    // safety check
    if (!systemmatrix_->filled())
      FOUR_C_THROW("Complete() has not been called on global system matrix yet!");

    // perform finite difference check on time integrator level
    if (scatra_field()->fd_check_type() == Inpar::ScaTra::fdcheck_global) fd_check();

    // check termination criterion for Newton-Raphson iteration
    if (exit_newton_raphson()) break;

    // initialize global increment vector
    increment_->put_scalar(0.);

    // store time before solving global system of equations
    time = timer_->wallTime();

    // equilibrate global system of equations if necessary
    equilibration_->equilibrate_system(systemmatrix_, residual_, blockmaps_);

    // solve global system of equations
    // Dirichlet boundary conditions have already been applied to global system of equations
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = iter_ == 1;
    solver_->solve(systemmatrix_, increment_, residual_, solver_params);

    equilibration_->unequilibrate_increment(increment_);

    // determine time needed for solving global system of equations,
    // and take maximum over all processors via communication
    double mydtsolve = timer_->wallTime() - time;
    dtsolve_ = Core::Communication::max_all(mydtsolve, get_comm());

    // output performance statistics associated with linear solver into text file if applicable
    if (fieldparameters_->get<bool>("OUTPUTLINSOLVERSTATS"))
      scatra_field()->output_lin_solver_stats(*solver_, dtsolve_, step(), static_cast<int>(iter_),
          residual_->get_map().num_global_elements());

    // update scatra field
    scatra_field()->update_iter(*maps_->extract_vector(*increment_, 0));
    scatra_field()->compute_intermediate_values();

    // update thermo field
    std::shared_ptr<Core::LinAlg::Vector<double>> thermoincrement(nullptr);
    if (condensationthermo_)
    {
      thermoincrement = std::make_shared<Core::LinAlg::Vector<double>>(
          *thermo_field()->discretization()->dof_row_map());
      Core::LinAlg::export_to(*maps_->extract_vector(*increment_, 1), *thermoincrement);
      const std::shared_ptr<const Core::LinAlg::Vector<double>> masterincrement =
          strategythermo_->interface_maps()->extract_vector(*thermoincrement, 2);
      const std::shared_ptr<Core::LinAlg::Vector<double>> slaveincrement =
          Core::LinAlg::create_vector(*strategythermo_->interface_maps()->map(1));
      switch (strategyscatra_->coupling_type())
      {
        case Inpar::S2I::coupling_matching_nodes:
        {
          icoupthermo_->master_to_slave(*masterincrement, *slaveincrement);
          break;
        }
        case Inpar::S2I::coupling_mortar_standard:
        {
          strategythermo_->p()->multiply(false, *masterincrement, *slaveincrement);
          break;
        }
        default:
        {
          FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
          break;
        }
      }
      strategythermo_->interface_maps()->insert_vector(*slaveincrement, 1, *thermoincrement);
    }
    else
    {
      thermoincrement = maps_->extract_vector(*increment_, 1);
    }
    thermo_field()->update_iter(*thermoincrement);
    thermo_field()->compute_intermediate_values();
  }  // Newton-Raphson iteration
}  // STI::Monolithic::Solve

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::Monolithic::apply_dirichlet_off_diag(
    std::shared_ptr<Core::LinAlg::SparseOperator> scatrathermo_domain_interface,
    std::shared_ptr<Core::LinAlg::SparseOperator> thermoscatra_domain_interface) const
{
  // apply Dirichlet boundary conditions to scatra-thermo matrix block
  scatrathermo_domain_interface->apply_dirichlet(*scatra_field()->dirich_maps()->cond_map(), false);

  // apply Dirichlet boundary conditions to scatra-thermo matrix block
  thermoscatra_domain_interface->apply_dirichlet(*thermo_field()->dirich_maps()->cond_map(), false);

  // zero out slave-side rows of thermo-scatra matrix block after having added them to the
  // corresponding master-side rows to finalize condensation of slave-side thermo dofs
  if (thermo_field()->s2i_meshtying())
  {
    switch (strategythermo_->coupling_type())
    {
      case Inpar::S2I::coupling_matching_nodes:
      {
        if (!thermo_field()->discretization()->has_condition("PointCoupling"))
          thermoscatra_domain_interface->apply_dirichlet(*icoupthermo_->slave_dof_map(), false);
        break;
      }

      case Inpar::S2I::coupling_mortar_condensed_bubnov:
      {
        thermoscatra_domain_interface->apply_dirichlet(
            *(strategythermo_->interface_maps()->map(1)), false);
        break;
      }

      default:
      {
        FOUR_C_THROW("Invalid type of scatra-scatra interface coupling!");
        break;
      }
    }
  }
}  // STI::Monolithic::ApplyDirichletOD

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void STI::Monolithic::assemble_domain_interface_off_diag(
    std::shared_ptr<Core::LinAlg::SparseOperator>& scatrathermo_domain_interface,
    std::shared_ptr<Core::LinAlg::SparseOperator>& thermoscatra_domain_interface) const
{
  // initialize scatra-thermo blocks
  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrathermo_domain_interface = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *blockmapthermo_, *scatra_field()->dof_block_maps(), 81, false, true);
      thermoscatra_domain_interface = std::make_shared<
          Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          *scatra_field()->dof_block_maps(), *blockmapthermo_, 81, false, true);

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      scatrathermo_domain_interface = std::make_shared<Core::LinAlg::SparseMatrix>(
          *scatra_field()->discretization()->dof_row_map(), 27, false, true);
      thermoscatra_domain_interface = std::make_shared<Core::LinAlg::SparseMatrix>(
          *thermo_field()->discretization()->dof_row_map(), 27, false, true);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown matrix type");
    }
  }

  // add domain and interface contibutions
  scatrathermo_domain_interface->add(*scatrathermoblockdomain_, false, 1.0, 1.0);
  scatrathermo_domain_interface->add(*scatrathermoblockinterface_, false, 1.0, 1.0);
  thermoscatra_domain_interface->add(*thermoscatrablockdomain_, false, 1.0, 1.0);
  thermoscatra_domain_interface->add(*thermoscatrablockinterface_, false, 1.0, 1.0);

  if (strategythermo_->coupling_type() == Inpar::S2I::coupling_matching_nodes)
  {  // standard meshtying algorithm with Lagrange multipliers condensed out
    if (!thermo_field()->discretization()->has_condition("PointCoupling"))
    {
      // during the very first run of the following code, Complete() has not yet been called
      // on the thermo-scatra block of the global system matrix experiments have shown that
      // the ExtractMatrixRows routine called in the following does not properly work in this
      // case if the thermo-scatra block exhibits a block structure in particular, some matrix
      // entries are simply omitted during extraction, although the underlying routine
      // ExtractGlobalRowCopy returns a zero error code as a consequence, the final
      // thermo-scatra block lacks some entries, and so does the final matrix graph after
      // calling Complete() this leads to errors involving unknown global indices of matrix
      // columns during the second run of the following code, with the thermo-scatra block now
      // being Complete() to circumvent this problem, we call Complete() on the thermo-scatra
      // block before the very first run of the following code
      if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::block_condition and
          step() == 1 and iter_ == 1)
        thermoscatra_domain_interface->complete();

      // loop over all thermo-scatra matrix blocks
      for (int iblock = 0; iblock < blockmaps_->num_maps() - 1; ++iblock)
      {
        // initialize temporary matrix for slave-side rows of current thermo-scatra matrix
        // block
        Core::LinAlg::SparseMatrix thermoscatrarowsslave(
            *icoupthermo_->slave_dof_map(), 27, false, true);

        // extract current thermo-scatra matrix block
        auto& thermoscatrablock =
            scatra_field()->matrix_type() == Core::LinAlg::MatrixType::block_condition
                ? std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
                      thermoscatra_domain_interface)
                      ->matrix(0, iblock)
                : *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
                      thermoscatra_domain_interface);

        // extract slave-side rows of thermo-scatra matrix block into temporary matrix
        ScaTra::MeshtyingStrategyS2I::extract_matrix_rows(
            thermoscatrablock, thermoscatrarowsslave, *icoupthermo_->slave_dof_map());

        // finalize temporary matrix with slave-side rows of thermo-scatra matrix block
        thermoscatrarowsslave.complete(*maps_->map(0), *icoupthermo_->slave_dof_map());

        // undo Complete() from above before performing subsequent matrix row transformation
        if (scatra_field()->matrix_type() == Core::LinAlg::MatrixType::block_condition and
            step() == 1 and iter_ == 1)
          thermoscatrablock.un_complete();

        // add slave-side rows of thermo-scatra matrix block to corresponding slave-side rows
        const std::shared_ptr<Coupling::Adapter::MatrixRowTransform>
            islavetomasterrowtransformthermood =
                scatra_field()->matrix_type() == Core::LinAlg::MatrixType::block_condition
                    ? std::make_shared<Coupling::Adapter::MatrixRowTransform>()
                    : islavetomasterrowtransformthermood_;
        (*islavetomasterrowtransformthermood)(thermoscatrarowsslave, 1.,
            Coupling::Adapter::CouplingSlaveConverter(*icoupthermo_), thermoscatrablock, true);
      }
    }
  }
  else if (strategythermo_->coupling_type() == Inpar::S2I::coupling_mortar_condensed_bubnov)
  {
    // standard meshtying algorithm with Lagrange multipliers condensed out
    // loop over all thermo-scatra matrix blocks
    for (int iblock = 0; iblock < blockmaps_->num_maps() - 1; ++iblock)
    {
      // initialize temporary matrix for slave-side rows of current thermo-scatra matrix block
      Core::LinAlg::SparseMatrix thermoscatrarowsslave(
          *strategythermo_->interface_maps()->map(1), 27, false, true);

      // extract current thermo-scatra matrix block
      auto& thermoscatrablock =
          scatra_field()->matrix_type() == Core::LinAlg::MatrixType::block_condition
              ? std::dynamic_pointer_cast<Core::LinAlg::BlockSparseMatrixBase>(
                    thermoscatra_domain_interface)
                    ->matrix(0, iblock)
              : *std::dynamic_pointer_cast<Core::LinAlg::SparseMatrix>(
                    thermoscatra_domain_interface);

      // extract slave-side rows of thermo-scatra matrix block into temporary matrix
      ScaTra::MeshtyingStrategyS2I::extract_matrix_rows(
          thermoscatrablock, thermoscatrarowsslave, *strategythermo_->interface_maps()->map(1));

      // finalize temporary matrix with slave-side rows of thermo-scatra matrix block
      thermoscatrarowsslave.complete(*maps_->map(0), *strategythermo_->interface_maps()->map(1));

      // add projected slave-side rows of thermo-scatra matrix block to corresponding
      // master-side rows
      thermoscatrablock.add(*Core::LinAlg::matrix_multiply(*strategythermo_->p(), true,
                                thermoscatrarowsslave, false, false, false, true),
          false, 1.0, 1.0);
    }
  }

  switch (scatra_field()->matrix_type())
  {
    case Core::LinAlg::MatrixType::block_condition:
    {
      scatrathermo_domain_interface->complete();
      thermoscatra_domain_interface->complete();

      break;
    }
    case Core::LinAlg::MatrixType::sparse:
    {
      scatrathermo_domain_interface->complete(
          *thermo_field()->discretization()->dof_row_map(), *maps_->map(0));
      thermoscatra_domain_interface->complete(*maps_->map(0), *maps_->map(1));

      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown matrix type");
      break;
    }
  }
}  // AssembleDomainInterfaceOD

FOUR_C_NAMESPACE_CLOSE
