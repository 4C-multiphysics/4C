// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_amgnxn_preconditioner.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linear_solver_amgnxn_vcycle.hpp"
#include "4C_utils_exceptions.hpp"

#include <MueLu_ParameterListInterpreter.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <filesystem>
#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AmGnxnPreconditioner::AmGnxnPreconditioner(Teuchos::ParameterList& params)
    : params_(params)
{
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::shared_ptr<Epetra_Operator> Core::LinearSolver::AmGnxnPreconditioner::prec_operator() const
{
  return p_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AmGnxnPreconditioner::setup(Core::LinAlg::SparseOperator& matrix,
    const Core::LinAlg::MultiVector<double>& x, Core::LinAlg::MultiVector<double>& b)
{
  // Check whether this is a block sparse matrix
  auto* A_bl = dynamic_cast<Core::LinAlg::BlockSparseMatrixBase*>(&matrix);
  if (A_bl == nullptr)
    FOUR_C_THROW(
        "The AMGnxn preconditioner works only for BlockSparseMatrixBase or derived classes");

  // Do all the setup
  setup(Core::Utils::shared_ptr_from_ref(*A_bl));
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AmGnxnPreconditioner::setup(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A)
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGnxn_Preconditioner::Setup");

  Teuchos::Time timer("", true);
  timer.reset();

  // Free old matrix and preconditioner
  a_ = nullptr;
  p_ = nullptr;

  // Create own copy of the system matrix in order to allow reusing the preconditioner
  a_ = A;
  a_ = a_->clone(Core::LinAlg::DataAccess::Copy);
  a_->complete();

  // Determine number of blocks
  int NumBlocks = a_->rows();
  if (a_->rows() != a_->cols())
    FOUR_C_THROW("The AMGnxn preconditioner works only for block square matrices");

  // Pick-up the input parameters
  AmGnxnInterface myInterface(params_, NumBlocks);

  // Create the Operator
  if (myInterface.get_preconditioner_type() == "AMG(BlockSmoother)")
  {
    p_ = std::make_shared<AmGnxnOperator>(a_, myInterface.get_num_pdes(),
        myInterface.get_null_spaces_dim(), myInterface.get_null_spaces_data(),
        myInterface.get_preconditioner_params(), myInterface.get_smoothers_params(),
        myInterface.get_smoothers_params());
  }
  else if (myInterface.get_preconditioner_type() == "BlockSmoother(X)")
  {
    p_ = std::make_shared<BlockSmootherOperator>(a_, myInterface.get_num_pdes(),
        myInterface.get_null_spaces_dim(), myInterface.get_null_spaces_data(),
        myInterface.get_preconditioner_params(), myInterface.get_smoothers_params());
  }
  else
    FOUR_C_THROW("Unknown preconditioner type: {}", myInterface.get_preconditioner_type());

  double elaptime = timer.totalElapsedTime(true);
  if (myInterface.get_preconditioner_params().get<std::string>("verbosity", "off") == "on" and
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(A->Comm())) == 0)
    std::cout << "       Calling Core::LinAlg::SOLVER::AMGnxn_Preconditioner::Setup takes "
              << std::setw(16) << std::setprecision(6) << elaptime << " s" << std::endl;

  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AmGnxnInterface::AmGnxnInterface(Teuchos::ParameterList& params, int NumBlocks)
{
  // Expected parameters in params
  //<ParameterList name="params">
  //
  //  <ParameterList name="AMGnxn Parameters">
  //    <Parameter name="AMGNXN_TYPE"          type="string"  value="AMG(BGS)"/>
  //    <!-- or -->
  //    <Parameter name="AMGNXN_TYPE"          type="string"  value="XML"/>
  //    <Parameter name="AMGNXN_XML_FILE"      type="string"  value=" ... .xml"/>
  //  </ParameterList>
  //
  //  <ParameterList name="Inverse1">
  //    <ParameterList name="MueLu Parameters">
  //    <Parameter name="PDE equations"          type="int"     value="..."/>
  //    <Parameter name="null space: dimension"  type="int"     value="..."/>
  //    <Parameter name="nullspace"              type="..."     value="..."/>
  //    </ParameterList>
  //  </ParameterList>
  //
  //</ParameterList>
  //
  //
  //
  // The xml file given in AMGNXN_XML_FILE should have the following format
  //<ParameterList name="dummy list which wraps everything">
  //
  //  <!-- Here we select which preconditioner we are going to use -->
  //  <Parameter name="Preconditioner"    type="string"  value="myPreconditioner"/>
  //
  //  <!-- Here we define our preconditioner -->
  //  <ParameterList name="myPreconditioner">
  //    <Parameter name="type"   type="string"  value="AMG(BlockSmoother)"/>
  //    <!-- or -- >
  //    <Parameter name="type"   type="string"  value="BlockSmoother(X)"/>
  //    <ParameterList name="parameters">
  //
  //     <!-- Fill this list with the parameters expected by your preconditioner -->
  //
  //    </ParameterList>
  //  </ParameterList>
  //
  //
  //  <!-- Here we put as many list as you need to define your smoothers -->
  //  <ParameterList name="mySmoother">
  //
  //     <!-- Fill this list with the parameters expected by your smoother -->
  //
  //  </ParameterList>
  //
  //</ParameterList>

  if (!params.isSublist("AMGnxn Parameters")) FOUR_C_THROW("AMGnxn Parameters not found!");
  Teuchos::ParameterList& amglist = params.sublist("AMGnxn Parameters");


  // Decide whether to choose a default type or parse a xml file
  std::string amgnxn_type = amglist.get<std::string>("AMGNXN_TYPE", "AMG(BGS)");
  if (amgnxn_type == "XML")
  {
    // Parse the whole file
    auto amgnxn_xml = amglist.get<std::optional<std::filesystem::path>>("AMGNXN_XML_FILE");
    if (!amgnxn_xml) FOUR_C_THROW("The input parameter AMGNXN_XML_FILE is 'none'.");
    {
      Teuchos::updateParametersFromXmlFile(
          amgnxn_xml->string(), Teuchos::Ptr<Teuchos::ParameterList>(&smoo_params_));

      // Find the path to the main xml file and include it in the param list to further usage
      smoo_params_.set<std::string>(
          "main xml path", amgnxn_xml->has_parent_path() ? amgnxn_xml->parent_path().string() : "");
    }
  }
  else
    FOUR_C_THROW(
        "\"{}\" is an invalid value for \"AMGNXN_TYPE\". Fix your input file", amgnxn_type);



  // Find preconditioner type and parameters
  std::string myprec = smoo_params_.get<std::string>("Preconditioner", "none");
  if (myprec == "none") FOUR_C_THROW("Not found \"Preconditioner\" parameter in your xml file.");
  if (!smoo_params_.isSublist(myprec))
    FOUR_C_THROW("Not found your preconditioner list in your xml file.");
  Teuchos::ParameterList& myprec_list = smoo_params_.sublist(myprec);
  prec_type_ = myprec_list.get<std::string>("type", "none");
  if (!myprec_list.isSublist("parameters"))
    FOUR_C_THROW("Not found the parameters list for your preconditioner. Fix your xml file.");
  prec_params_ = myprec_list.sublist("parameters");

  // Find null spaces and relatives
  std::string Inverse_str = "Inverse";
  xml_files_.resize(NumBlocks);
  num_pdes_.resize(NumBlocks);
  null_spaces_dim_.resize(NumBlocks);
  null_spaces_data_.resize(NumBlocks);
  for (int block = 0; block < NumBlocks; block++)
  {
    if (!params.isSublist(Inverse_str + convert_int(block + 1)))
      FOUR_C_THROW("Not found inverse list for block {}", block + 1);
    Teuchos::ParameterList& inverse_list = params.sublist(Inverse_str + convert_int(block + 1));

    if (!inverse_list.isSublist("MueLu Parameters")) FOUR_C_THROW("MueLu Parameters not found");
    Teuchos::ParameterList& mllist = inverse_list.sublist("MueLu Parameters");

    xml_files_[block] = mllist.get<std::string>("xml file", "none");
    num_pdes_[block] = mllist.get<int>("PDE equations", -1);
    null_spaces_dim_[block] = mllist.get<int>("null space: dimension", -1);

    std::shared_ptr<Core::LinAlg::MultiVector<double>> nullspace =
        mllist.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("nullspace", nullptr);
    if (nullspace == nullptr) FOUR_C_THROW("Nullspace vector is null!");

    std::shared_ptr<std::vector<double>> ns =
        std::make_shared<std::vector<double>>(nullspace->MyLength() * nullspace->NumVectors());

    Core::LinAlg::multi_vector_to_std_vector(*nullspace, *ns, null_spaces_dim_[block]);
    null_spaces_data_[block] = ns;

    // Some checks
    if (num_pdes_[block] < 1 or null_spaces_dim_[block] < 1)
      FOUR_C_THROW("Error: PDE equations or null space dimension wrong.");
    if (null_spaces_data_[block] == nullptr) FOUR_C_THROW("Error: null space data is empty");
  }
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AmGnxnOperator::AmGnxnOperator(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A, std::vector<int> num_pdes,
    std::vector<int> null_spaces_dim,
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data,
    const Teuchos::ParameterList& amgnxn_params, const Teuchos::ParameterList& smoothers_params,
    const Teuchos::ParameterList& muelu_params)
    : a_(A),
      num_pdes_(num_pdes),
      null_spaces_dim_(null_spaces_dim),
      null_spaces_data_(null_spaces_data),
      amgnxn_params_(amgnxn_params),
      smoothers_params_(smoothers_params),
      muelu_params_(muelu_params),
      is_setup_flag_(false)
{
  // Expected parameters in amgnxn_params (example)
  //<ParameterList name="amgnxn_params">
  //
  //  <Parameter name="number of levels"                 type="int"  value="..."/>
  //
  //  <Parameter name="smoother: all but coarsest level" type="string"  value="myFinestSmoother"/>
  //
  //  <Parameter name="smoother: coarsest level"         type="string"  value="myCoarsestSmoother"/>
  //
  //  <Parameter name="verbosity"                        type="string"  value="on"/>
  //
  //  <Parameter name="muelu parameters for block 0"       type="string"  value="myMuelu0"/>
  //
  //  <Parameter name="muelu parameters for block 1"       type="string"  value="myMuelu1"/>
  //
  //   ....
  //
  //  <Parameter name="muelu parameters for block N"       type="string"  value="myMueluN"/>
  //
  //</ParameterList>

  // Expected parameters in smoothers_params (example)
  //<ParameterList name="smoothers_params">
  //
  //  <ParameterList name="myFinestSmoother">
  //
  //   ...    ...    ...    ...    ...
  //
  //  </ParameterList>
  //
  //  <ParameterList name="myCoarsestSmoother">
  //
  //   ...    ...    ...    ...    ...
  //
  //  </ParameterList>
  //
  //</ParameterList>

  // Expected parameters in muelu_params (example)
  //<ParameterList name="muelu_params">
  //
  //   <ParameterList name="myMueluX">
  //     <Parameter name="xml file"      type="string"  value="myfile.xml"/>
  //   </ParameterList>
  //
  //   TODO or
  //
  //   <ParameterList name="myMueluX">
  //    ... ... list defining the muelue hierarchy (i.e.) the contents of the xml file
  //   </ParameterList>
  //
  //
  //</ParameterList>

  setup();
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::AmGnxnOperator::ApplyInverse(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGnxn_Operator::ApplyInverse");
  if (!is_setup_flag_)
    FOUR_C_THROW("ApplyInverse cannot be called without a previous set up of the preconditioner");

  const Core::LinAlg::MultiMapExtractor& range_ex = a_->range_extractor();
  const Core::LinAlg::MultiMapExtractor& domain_ex = a_->domain_extractor();

  int NumBlocks = a_->rows();
  if (NumBlocks != a_->cols()) FOUR_C_THROW("The block matrix has to be square");

  AMGNxN::BlockedVector Xbl(NumBlocks);
  AMGNxN::BlockedVector Ybl(NumBlocks);

  int NV = X.NumVectors();
  for (int i = 0; i < NumBlocks; i++)
  {
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Xi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(range_ex.map(i)), NV);

    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Yi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(domain_ex.map(i)), NV);

    range_ex.extract_vector(Core::LinAlg::MultiVector<double>(X), i, *Xi);
    domain_ex.extract_vector(Core::LinAlg::MultiVector<double>(X), i, *Yi);
    Xbl.set_vector(Xi, i);
    Ybl.set_vector(Yi, i);
  }

  if (v_ == nullptr) FOUR_C_THROW("Null pointer. We cannot call the vcycle");

  v_->solve(Xbl, Ybl, true);

  Core::LinAlg::View Y_view(Y);
  for (int i = 0; i < NumBlocks; i++) domain_ex.insert_vector(*(Ybl.get_vector(i)), i, Y_view);

  return 0;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AmGnxnOperator::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGnxn_Operator::Setup");


  int NumBlocks = a_->rows();
  if (NumBlocks != a_->cols()) FOUR_C_THROW("We spect a square matrix here");

  // Extract the blockedMatrix
  Teuchos::RCP<AMGNxN::BlockedMatrix> Able =
      Teuchos::make_rcp<AMGNxN::BlockedMatrix>(NumBlocks, NumBlocks);
  for (int i = 0; i < NumBlocks; i++)
  {
    for (int j = 0; j < NumBlocks; j++)
    {
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Aij = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(
          a_->matrix(i, j), Core::LinAlg::DataAccess::Share);
      Able->set_matrix(Aij, i, j);
    }
  }


  v_ = std::make_shared<AMGNxN::CoupledAmg>(Able, num_pdes_, null_spaces_dim_, null_spaces_data_,
      amgnxn_params_, smoothers_params_, muelu_params_);


  is_setup_flag_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::BlockSmootherOperator::BlockSmootherOperator(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A, std::vector<int> num_pdes,
    std::vector<int> null_spaces_dim,
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data,
    const Teuchos::ParameterList& amgnxn_params, const Teuchos::ParameterList& smoothers_params)
    : a_(A),
      num_pdes_(num_pdes),
      null_spaces_dim_(null_spaces_dim),
      null_spaces_data_(null_spaces_data),
      amgnxn_params_(amgnxn_params),
      smoothers_params_(smoothers_params),
      is_setup_flag_(false)
{
  // Expected parameters in amgnxn_params (example)
  //<ParameterList name="amgnxn_params">
  //
  //  <Parameter name="smoother"         type="string"  value="myBlockSmoother"/>
  //
  //  <Parameter name="verbosity"        type="string"  value="on"/>
  //
  //</ParameterList>

  // Expected parameters in smoothers_params (example)
  //<ParameterList name="smoothers_params">
  //
  //  <ParameterList name="myBlockSmoother">
  //
  //   ...    ...    ...    ...    ...
  //
  //  </ParameterList>
  //
  //
  //</ParameterList>

  setup();
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::BlockSmootherOperator::ApplyInverse(
    const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::BlockSmoother_Operator::ApplyInverse");

  if (!is_setup_flag_)
    FOUR_C_THROW("ApplyInverse cannot be called without a previous set up of the preconditioner");


  const Core::LinAlg::MultiMapExtractor& range_ex = a_->range_extractor();
  const Core::LinAlg::MultiMapExtractor& domain_ex = a_->domain_extractor();

  int NumBlocks = a_->rows();
  if (NumBlocks != a_->cols()) FOUR_C_THROW("The block matrix has to be square");

  AMGNxN::BlockedVector Xbl(NumBlocks);
  AMGNxN::BlockedVector Ybl(NumBlocks);
  int NV = X.NumVectors();
  for (int i = 0; i < NumBlocks; i++)
  {
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Xi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(range_ex.map(i)), NV);
    Teuchos::RCP<Core::LinAlg::MultiVector<double>> Yi =
        Teuchos::make_rcp<Core::LinAlg::MultiVector<double>>(*(domain_ex.map(i)), NV);
    range_ex.extract_vector(Core::LinAlg::MultiVector<double>(X), i, *Xi);
    domain_ex.extract_vector(Core::LinAlg::MultiVector<double>(X), i, *Yi);
    Xbl.set_vector(Xi, i);
    Ybl.set_vector(Yi, i);
  }


  if (s_ == Teuchos::null) FOUR_C_THROW("Null pointer. We cannot call the smoother");

  s_->solve(Xbl, Ybl, true);

  Core::LinAlg::View Y_view(Y);
  for (int i = 0; i < NumBlocks; i++) domain_ex.insert_vector(*(Ybl.get_vector(i)), i, Y_view);



  return 0;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::BlockSmootherOperator::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::BlockSmoother_Operator::Setup");


  std::string verbosity = amgnxn_params_.get<std::string>("verbosity", "off");

  if (Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(a_->Comm())) != 0)
    verbosity = "off";



  if (verbosity == "on")
  {
    std::cout << "===============================================" << std::endl;
    std::cout << "Core::LinAlg::SOLVER::BlockSmoother_Operator : debug info  (begin)" << std::endl;
    std::cout << std::endl;
  }


  // Prepare info
  int NumBlocks = a_->rows();
  std::string smother_name = amgnxn_params_.get<std::string>("smoother", "BGS");
  std::vector<int> blocks(NumBlocks, 0);
  for (int i = 0; i < NumBlocks; i++) blocks[i] = i;
  std::vector<AMGNxN::NullSpaceInfo> null_space_blocks;
  for (int i = 0; i < NumBlocks; i++)
  {
    AMGNxN::NullSpaceInfo myNS(num_pdes_[i], null_spaces_dim_[i], null_spaces_data_[i]);
    null_space_blocks.push_back(myNS);
  }
  Teuchos::RCP<AMGNxN::BlockedMatrix> Able =
      Teuchos::make_rcp<AMGNxN::BlockedMatrix>(NumBlocks, NumBlocks);
  for (int i = 0; i < NumBlocks; i++)
  {
    for (int j = 0; j < NumBlocks; j++)
    {
      Teuchos::RCP<Core::LinAlg::SparseMatrix> Aij = Teuchos::make_rcp<Core::LinAlg::SparseMatrix>(
          a_->matrix(i, j), Core::LinAlg::DataAccess::Share);
      Able->set_matrix(Aij, i, j);
    }
  }

  // smoother factory
  AMGNxN::SmootherFactory mySmootherCreator;
  mySmootherCreator.set_operator(Able);
  mySmootherCreator.set_params_smoother(smoothers_params_);
  mySmootherCreator.set_level(0);
  mySmootherCreator.set_blocks(blocks);
  mySmootherCreator.set_smoother_name(smother_name);
  mySmootherCreator.set_verbosity(verbosity);
  mySmootherCreator.set_null_space_all_blocks(null_space_blocks);

  // Create smoother
  sbase_ = mySmootherCreator.create();
  s_ = Teuchos::rcp_dynamic_cast<AMGNxN::BlockedSmoother>(sbase_);
  if (s_ == Teuchos::null)
    FOUR_C_THROW("We expect a blocked smoother. Fix the xml file defining the smoother");

  //// Print maps
  // for(int i=0;i<NumBlocks;i++)
  //{
  //  std::stringstream sstr;
  //  sstr << "Map_block" << i;
  //  PrintMap(A_->Matrix(i,i).RowMap(),sstr.str());
  //}


  if (verbosity == "on")
  {
    std::cout << std::endl;
    std::cout << "Core::LinAlg::SOLVER::BlockSmoother_Operator : debug info  (end)" << std::endl;
    std::cout << "===============================================" << std::endl;
  }

  is_setup_flag_ = true;
  return;
}

FOUR_C_NAMESPACE_CLOSE
