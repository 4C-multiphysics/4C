// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linear_solver_amgnxn_smoothers.hpp"

#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.hpp"

#include <MueLu_EpetraOperator.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::GenericSmoother::richardson(GenericSmoother& Ainv,
    const BlockedMatrix& A, const BlockedVector& X, BlockedVector& Y, int iters, double omega,
    bool InitialGuessIsZero) const
{
  BlockedVector DX = X.deep_copy();
  BlockedVector DY = Y.deep_copy();  // TODO we only need a new vector

  for (int i = 0; i < iters; i++)
  {
    if (i != 0 or not InitialGuessIsZero)
    {
      A.apply(Y, DX);
      DX.update(1.0, X, -1.0);
    }

    // DY.PutScalar(0.0);
    Ainv.solve(DX, DY, true);

    if (i != 0 or not InitialGuessIsZero)
      Y.update(omega, DY, 1.0);
    else
      Y.update(omega, DY, 0.0);
  }
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::BgsSmoother::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::AMGNxN::BgsSmoother::Solve");

  unsigned NumSuperBlocks = superblocks_.size();

  for (unsigned k = 0; k < iter_; k++)
  {
    for (unsigned i = 0; i < NumSuperBlocks; i++)
    {
      BlockedVector DXi = X.get_blocked_vector(superblocks_[i]).deep_copy();
      BlockedVector DXitmp = DXi.deep_copy();  // TODO we only need a new vector
      for (unsigned j = 0; j < NumSuperBlocks; j++)
      {
        if (k != 0 or not InitialGuessIsZero or j < i)
        {
          BlockedVector Yj = Y.get_blocked_vector(superblocks_[j]);
          BlockedMatrix Aij = a_->get_blocked_matrix(superblocks_[i], superblocks_[j]);
          Aij.apply(Yj, DXitmp);
          DXi.update(-1.0, DXitmp, 1.0);
        }
      }

      BlockedVector Yi = Y.get_blocked_vector(superblocks_[i]);
      BlockedVector DYi = Yi.deep_copy();  // TODO we only need a new vector
      BlockedMatrix Aii = a_->get_blocked_matrix(superblocks_[i], superblocks_[i]);
      richardson(*smoothers_[i], Aii, DXi, DYi, iters_[i], omegas_[i], true);

      if (k != 0 or not InitialGuessIsZero)
        Yi.update(omega_, DYi, 1.0);
      else
        Yi.update(omega_, DYi, 0.0);
    }
  }
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::MueluAMGWrapper::MueluAMGWrapper(
    Teuchos::RCP<Core::LinAlg::SparseMatrix> A, int num_pde, int null_space_dim,
    std::shared_ptr<std::vector<double>> null_space_data, const Teuchos::ParameterList& muelu_list)
    : A_(std::move(A)),
      num_pde_(num_pde),
      null_space_dim_(null_space_dim),
      null_space_data_(std::move(null_space_data)),
      muelu_list_(muelu_list)
{
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::build_hierarchy()
{
  // Prepare operator for MueLu
  auto A_crs = Teuchos::rcpFromRef(A_->epetra_matrix());
  if (A_crs == Teuchos::null)
    FOUR_C_THROW("Make sure that the input matrix is a Epetra_CrsMatrix (or derived)");
  Teuchos::RCP<Xpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluA =
      Teuchos::make_rcp<Xpetra::EpetraCrsMatrixT<int, Xpetra::EpetraNode>>(A_crs);

  Teuchos::RCP<Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluA_wrap =
      Teuchos::make_rcp<Xpetra::CrsMatrixWrap<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(mueluA);
  Teuchos::RCP<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> mueluOp =
      Teuchos::rcp_dynamic_cast<Xpetra::Matrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>>(
          mueluA_wrap);

  // Prepare null space vector for MueLu
  // safety check
  const size_t localNumRows = mueluA->getLocalNumRows();

  if (localNumRows * null_space_dim_ != null_space_data_->size())
    FOUR_C_THROW("Matrix size is inconsistent with length of nullspace vector!");
  Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> rowMap = mueluA->getRowMap();
  Teuchos::RCP<Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>> nspVector =
      Xpetra::MultiVectorFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(
          rowMap, null_space_dim_, true);
  for (size_t i = 0; i < Teuchos::as<size_t>(null_space_dim_); i++)
  {
    Teuchos::ArrayRCP<Scalar> nspVectori = nspVector->getDataNonConst(i);
    const size_t myLength = nspVector->getLocalLength();
    for (size_t j = 0; j < myLength; j++)
    {
      nspVectori[j] = (*null_space_data_)[i * myLength + j];
    }
  }


  // Input num eq and offset in the final level.
  // The amalgamation factory needs this info!
  int offsetFineLevel(0);
  if (num_pde_ > 1) offsetFineLevel = A_->row_map().min_all_gid();
  mueluOp->SetFixedBlockSize(num_pde_, offsetFineLevel);
  Teuchos::ParameterList& MatrixList = muelu_list_.sublist("Matrix");
  MatrixList.set<int>("DOF offset", offsetFineLevel);
  MatrixList.set<int>("number of equations", num_pde_);
  // std::cout << "offsetFineLevel " << offsetFineLevel << std::endl;



  // Build up hierarchy
  MueLu::ParameterListInterpreter<Scalar, LocalOrdinal, GlobalOrdinal, Node> mueLuFactory(
      muelu_list_);
  H_ = mueLuFactory.CreateHierarchy();
  H_->SetDefaultVerbLevel(MueLu::Extreme);  // TODO sure?
  H_->GetLevel(0)->Set("A", mueluOp);
  H_->GetLevel(0)->Set("Nullspace", nspVector);
  H_->GetLevel(0)->setlib(Xpetra::UseEpetra);
  H_->setlib(Xpetra::UseEpetra);
  mueLuFactory.SetupHierarchy(*H_);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::setup()
{
  TEUCHOS_FUNC_TIME_MONITOR("Core::LinAlg::SOLVER::MueluAMGWrapper::Setup");

  Teuchos::Time timer("", true);
  timer.reset();

  // Create the hierarchy
  build_hierarchy();

  // Create the V-cycle
  p_ = Teuchos::make_rcp<MueLu::EpetraOperator>(H_);

  double elaptime = timer.totalElapsedTime(true);
  if (muelu_list_.sublist("Hierarchy").get<std::string>("verbosity", "None") != "None" and
      Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(A_->Comm())) == 0)
    std::cout << "       Calling Core::LinAlg::SOLVER::AMGNxN::MueluAMGWrapper::Setup takes "
              << std::setw(16) << std::setprecision(6) << elaptime << " s" << std::endl;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::MueluAMGWrapper::apply(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const
{
  if (InitialGuessIsZero)
    Y.put_scalar(0.0);  // TODO Remove when you are sure that ApplyInverse will zero out Y.
  p_->ApplyInverse(X.get_epetra_multi_vector(), Y.get_epetra_multi_vector());
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


Core::LinearSolver::AMGNxN::SmootherManager::SmootherManager()
    : set_operator_(false),
      set_params_(false),
      set_params_subsolver_(false),
      set_hierarchies_(false),
      set_level_(false),
      set_block_(false),
      set_blocks_(false),
      set_type_(false),
      set_verbosity_(false),
      set_null_space_(false),
      set_null_space_all_blocks_(false)
{
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::BlockedMatrix>
Core::LinearSolver::AMGNxN::SmootherManager::get_operator()
{
  return operator_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::ParameterList Core::LinearSolver::AMGNxN::SmootherManager::get_params() { return params_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::ParameterList Core::LinearSolver::AMGNxN::SmootherManager::get_params_smoother()
{
  return params_subsolver_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::Hierarchies>
Core::LinearSolver::AMGNxN::SmootherManager::get_hierarchies()
{
  return hierarchies_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::AMGNxN::SmootherManager::get_level() { return level_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

int Core::LinearSolver::AMGNxN::SmootherManager::get_block() { return block_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::vector<int> Core::LinearSolver::AMGNxN::SmootherManager::get_blocks() { return blocks_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_smoother_name()
{
  return subsolver_name_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_type() { return type_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::string Core::LinearSolver::AMGNxN::SmootherManager::get_verbosity() { return verbosity_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Core::LinearSolver::AMGNxN::NullSpaceInfo
Core::LinearSolver::AMGNxN::SmootherManager::get_null_space()
{
  return null_space_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

std::vector<Core::LinearSolver::AMGNxN::NullSpaceInfo>
Core::LinearSolver::AMGNxN::SmootherManager::get_null_space_all_blocks()
{
  return null_space_all_blocks_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_operator(Teuchos::RCP<BlockedMatrix> in)
{
  set_operator_ = true;
  operator_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_params(const Teuchos::ParameterList& in)
{
  set_params_ = true;
  params_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_params_smoother(
    const Teuchos::ParameterList& in)
{
  set_params_subsolver_ = true;
  params_subsolver_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_hierarchies(Teuchos::RCP<Hierarchies> in)
{
  set_hierarchies_ = true;
  hierarchies_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_level(int in)
{
  set_level_ = true;
  level_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_block(int in)
{
  set_block_ = true;
  block_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_blocks(std::vector<int> in)
{
  set_blocks_ = true;
  blocks_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_smoother_name(std::string in)
{
  set_subsolver_name_ = true;
  subsolver_name_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_type(std::string in)
{
  set_type_ = true;
  type_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_verbosity(std::string in)
{
  set_verbosity_ = true;
  verbosity_ = std::move(in);
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_null_space(const NullSpaceInfo& in)
{
  set_null_space_ = true;
  null_space_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherManager::set_null_space_all_blocks(
    const std::vector<NullSpaceInfo>& in)
{
  set_null_space_all_blocks_ = true;
  null_space_all_blocks_ = in;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_operator() { return set_operator_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_params() { return set_params_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_params_smoother()
{
  return set_params_subsolver_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_hierarchies() { return set_hierarchies_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_level() { return set_level_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_block() { return set_block_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_blocks() { return set_blocks_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_smoother_name()
{
  return set_subsolver_name_;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_type() { return set_type_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_verbosity() { return set_verbosity_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_null_space() { return set_null_space_; }

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

bool Core::LinearSolver::AMGNxN::SmootherManager::is_set_null_space_all_blocks()
{
  return set_null_space_all_blocks_;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::SmootherFactory::set_type_and_params()
{
  // Valid types
  std::vector<std::string> valid_types;
  valid_types.emplace_back("BGS");
  valid_types.emplace_back("IFPACK");
  valid_types.emplace_back("REUSE_MUELU_SMOOTHER");
  valid_types.emplace_back("REUSE_MUELU_AMG");
  valid_types.emplace_back("NEW_MUELU_AMG");
  valid_types.emplace_back("NEW_MUELU_AMG_IFPACK_SMO");
  valid_types.emplace_back("DIRECT_SOLVER");
  valid_types.emplace_back("MERGE_AND_SOLVE");
  valid_types.emplace_back("BLOCK_AMG");
  valid_types.emplace_back("SIMPLE");

  std::string smoother_type;
  Teuchos::ParameterList smoother_params;
  if (get_params_smoother().isSublist(get_smoother_name()))
  {
    smoother_type =
        get_params_smoother().sublist(get_smoother_name()).get<std::string>("type", "none");
    smoother_params = get_params_smoother().sublist(get_smoother_name()).sublist("parameters");
  }
  else if (std::find(valid_types.begin(), valid_types.end(), get_smoother_name()) !=
           valid_types.end())
    smoother_type = get_smoother_name();
  else
    smoother_type = "none";

  set_type(smoother_type);
  set_params(smoother_params);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::SmootherFactory::create()
{
  // Expected parameters in GetParamsSmoother()
  //
  // <ParameterList name="mySmoother">
  //   <Parameter name="type"   type="string"  value="..."/>
  //   <ParameterList name="parameters">
  //
  //    ...    ...   ...   ...   ...   ...
  //
  //   </ParameterList>
  // </ParameterList>
  //
  // Available input?

  if (not is_set_params_smoother()) FOUR_C_THROW("IsSetParamsSmoother() returns false");
  if (not is_set_smoother_name()) FOUR_C_THROW("IsSetSmootherName() returns false");

  // Determine the type of smoother to be constructed and its parameters
  set_type_and_params();

  // Create the corresponding factory
  Teuchos::RCP<SmootherFactoryBase> mySmootherFactory = Teuchos::null;

  if (get_type() == "BGS")
  {
    mySmootherFactory = Teuchos::make_rcp<BgsSmootherFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_blocks(get_blocks());
    mySmootherFactory->set_null_space_all_blocks(get_null_space_all_blocks());
  }
  else if (get_type() == "NEW_MUELU_AMG")
  {
    mySmootherFactory = Teuchos::make_rcp<MueluAMGWrapperFactory>();
    mySmootherFactory->set_operator(get_operator());
    mySmootherFactory->set_hierarchies(get_hierarchies());
    mySmootherFactory->set_block(get_block());
    mySmootherFactory->set_params(get_params());
    mySmootherFactory->set_params_smoother(get_params_smoother());
    mySmootherFactory->set_null_space(get_null_space());
  }
  else
    FOUR_C_THROW("Unknown smoother type. Fix your xml file");

  // Build the smoother
  mySmootherFactory->set_verbosity(get_verbosity());
  mySmootherFactory->set_level(get_level());
  return mySmootherFactory->create();
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::MueluAMGWrapperFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="xml file"      type="string"  value="myfile.xml"/>
  // </ParameterList>
  //
  // TODO or
  //
  // <ParameterList name="parameters">
  //   <Parameter name="parameter list"      type="string"  value="NameOfTheParameterList"/>
  // </ParameterList>
  //  In that case we look in GetParamsSmoother() for a list called "NameOfTheParameterList"
  //  which has to contain all the parameters defining a muelu hierarchy
  //
  // TODO or
  //
  // <ParameterList name="parameters">
  //  ... ... list defining the muelue hierarchy (i.e.) the contents of the xml file
  // </ParameterList>
  //
  //
  // Priority: first "xml file", then "parameter list", then other:
  // If the parameter "xml file" is found, then all other parameters are ignored
  // else, if "parameter list is found", then other parameters are ignored

  // Check input
  if (not is_set_level()) FOUR_C_THROW("IsSetLevel() returns false");
  if (not is_set_operator()) FOUR_C_THROW("IsSetOperator() returns false");
  if (not is_set_block()) FOUR_C_THROW("IsSetBlock() returns false");
  if (not is_set_hierarchies()) FOUR_C_THROW("IsSetHierarchies() returns false");
  if (not is_set_params()) FOUR_C_THROW("IsSetParams() returns false");
  if (not is_set_null_space()) FOUR_C_THROW("IsSetNullSpace() returns false");
  if (not is_set_params_smoother()) FOUR_C_THROW("IsSetSmoothersParams() returns false");

  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a NEW_MUELU_AMG smoother for block " << get_block();
    std::cout << " at level " << get_level() << std::endl;
  }

  Teuchos::ParameterList myList;
  std::string xml_filename = get_params().get<std::string>("xml file", "none");
  std::string list_name = get_params().get<std::string>("parameter list", "none");
  if (xml_filename != "none")
  {
    // If the xml file is not an absolute path, make it relative wrt the main xml file
    if ((xml_filename)[0] != '/')
    {
      std::string tmp = get_params_smoother().get<std::string>("main xml path", "none");
      if (tmp == "none") FOUR_C_THROW("Path of the main xml not found");
      xml_filename.insert(xml_filename.begin(), tmp.begin(), tmp.end());
    }

    Teuchos::updateParametersFromXmlFile(
        xml_filename, Teuchos::Ptr<Teuchos::ParameterList>(&myList));

    if (get_verbosity() == "on")
    {
      std::cout << "The chosen parameters are:" << std::endl;
      std::cout << "xml file = : " << xml_filename << std::endl;
    }
  }
  else if (list_name != "none")
  {
    myList = get_params_smoother().sublist(list_name);
    if (get_verbosity() == "on")
    {
      std::cout << "The chosen parameters are:" << std::endl;
      std::cout << "parameter list = : " << list_name << std::endl;
    }
  }
  else
    myList = get_params();



  // TODO now we use the null space generated by 4C, which only makes sense for the finest level.
  // We can obtain null spaces for other levels from inside the muelu hierarchies.
  if (get_level() != 0)
  {
    FOUR_C_THROW(
        "Trying to create a NEW_MUELU_AMG smoother at a level > 0. Sorry, but this is not possible "
        "yet.");
  }

  // Recover info
  if (not get_operator()->has_only_one_block())
    FOUR_C_THROW("This smoother can be built only for single block matrices");
  Teuchos::RCP<Core::LinAlg::SparseMatrix> Op2 = get_operator()->get_matrix(0, 0);
  if (Op2 == Teuchos::null) FOUR_C_THROW("I dont want a null pointer here");
  int num_pde = get_null_space().get_num_pd_es();
  int null_space_dim = get_null_space().get_null_space_dim();
  auto null_space_data = get_null_space().get_null_space_data();

  Teuchos::RCP<MueluAMGWrapper> PtrOut =
      Teuchos::make_rcp<MueluAMGWrapper>(Op2, num_pde, null_space_dim, null_space_data, myList);
  PtrOut->setup();

  return Teuchos::rcp_dynamic_cast<Core::LinearSolver::AMGNxN::GenericSmoother>(PtrOut);
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

Teuchos::RCP<Core::LinearSolver::AMGNxN::GenericSmoother>
Core::LinearSolver::AMGNxN::BgsSmootherFactory::create()
{
  // Expected parameters (example)
  // <ParameterList name="parameters">
  //   <Parameter name="blocks"      type="string"  value="(1,2),(3,4),(5)"/>
  //   <Parameter name="smoothers"   type="string"  value="myBGS,mySIMPLE,IFPACK"/>
  //   <Parameter name="sweeps"      type="int"     value="3"/>
  //   <Parameter name="omega"       type="double"  value="1.0"/>
  //   <Parameter name="local sweeps"  type="string"     value="3,2,4"/>
  //   <Parameter name="local omegas"  type="string"  value="1.0,1.2,0.9"/>
  // </ParameterList>

  // TODO Check that all required data is set

  // =============================================================
  // Parse parameters
  // =============================================================

  // determine how the blocks are grouped
  std::string blocks_string = get_params().get<std::string>("blocks", "none");
  std::vector<std::vector<int>> SuperBlocks2Blocks;
  std::vector<std::vector<int>> SuperBlocks2BlocksLocal;
  get_operator()->parse_blocks(
      blocks_string, get_blocks(), SuperBlocks2Blocks, SuperBlocks2BlocksLocal);

  // std::cout << "======================" << std::endl;
  // for(size_t i=0;i<SuperBlocks2Blocks.size();i++)
  // {
  //   for(size_t j=0;j<SuperBlocks2Blocks[i].size();j++)
  //     std::cout << SuperBlocks2Blocks[i][j] << ", ";
  //   std::cout << std::endl;
  // }


  // Determine the subsolver names
  std::string smoothers_string = get_params().get<std::string>("smoothers", "none");
  std::vector<std::string> SubSolverNames;
  parse_smoother_names(smoothers_string, SubSolverNames, SuperBlocks2Blocks);

  // sweeps and damping
  unsigned iter = static_cast<unsigned>(get_params().get<int>("sweeps", 1));
  double omega = get_params().get<double>("omega", 1.0);
  std::string local_sweeps = get_params().get<std::string>("local sweeps", "none");
  std::string local_omegas = get_params().get<std::string>("local omegas", "none");
  unsigned NumSuperBlocks = SuperBlocks2Blocks.size();
  std::vector<double> omegas(NumSuperBlocks, 1.0);
  std::vector<unsigned> iters(NumSuperBlocks, 1);
  if (local_sweeps != "none")
  {
    std::istringstream ss(local_sweeps);
    std::string token;
    unsigned ib = 0;
    while (std::getline(ss, token, ','))
    {
      if (ib >= NumSuperBlocks) FOUR_C_THROW("too many comas in {}", local_sweeps);
      iters[ib++] = atoi(token.c_str());
    }
    if (ib < NumSuperBlocks) FOUR_C_THROW("too less comas in {}", local_sweeps);
  }
  if (local_omegas != "none")
  {
    std::istringstream ss(local_omegas);
    std::string token;
    unsigned ib = 0;
    while (std::getline(ss, token, ','))
    {
      if (ib >= NumSuperBlocks) FOUR_C_THROW("too many comas in {}", local_omegas);
      omegas[ib++] = atof(token.c_str());
    }
    if (ib < NumSuperBlocks) FOUR_C_THROW("too less comas in {}", local_omegas);
  }



  // =============================================================
  // Some output
  // =============================================================
  if (get_verbosity() == "on")
  {
    std::cout << std::endl;
    std::cout << "Creating a BGS smoother for blocks (";
    for (size_t i = 0; i < get_blocks().size(); i++)
    {
      std::cout << get_blocks()[i];
      if (i < get_blocks().size() - 1)
        std::cout << ", ";
      else
        std::cout << ") ";
    }
    std::cout << " at level " << get_level() << std::endl;
    std::cout << "The chosen parameters are" << std::endl;
    std::cout << "blocks = ";
    for (size_t k = 0; k < SuperBlocks2Blocks.size(); k++)
    {
      std::cout << "(";
      for (size_t j = 0; j < SuperBlocks2Blocks[k].size(); j++)
      {
        std::cout << SuperBlocks2Blocks[k][j];
        if (j < (SuperBlocks2Blocks[k].size() - 1)) std::cout << ",";
      }
      if (k < (SuperBlocks2Blocks.size() - 1))
        std::cout << "),";
      else
        std::cout << ")" << std::endl;
    }
    std::cout << "smoothers = ";
    for (size_t k = 0; k < SubSolverNames.size(); k++)
    {
      std::cout << SubSolverNames[k];
      if (k < (SubSolverNames.size() - 1))
        std::cout << ",";
      else
        std::cout << std::endl;
    }
    std::cout << "sweeps = " << iter << std::endl;
    std::cout << "omega = " << omega << std::endl;
    std::cout << "local sweeps = ";
    for (int i : iters) std::cout << i << ",";
    std::cout << std::endl;
    std::cout << "local omegas = ";
    for (double o : omegas) std::cout << o << ",";
    std::cout << std::endl;
    // std::cout << std::endl;
  }


  // =============================================================
  // Construct smoothers for diagonal superblocks
  // =============================================================

  std::vector<Teuchos::RCP<GenericSmoother>> SubSmoothers(NumSuperBlocks, Teuchos::null);
  for (unsigned scol = 0; scol < NumSuperBlocks; scol++)
  {
    SmootherFactory mySmootherCreator;
    mySmootherCreator.set_smoother_name(SubSolverNames[scol]);
    mySmootherCreator.set_params_smoother(get_params_smoother());
    mySmootherCreator.set_hierarchies(get_hierarchies());
    mySmootherCreator.set_level(get_level());
    const std::vector<int>& scols = SuperBlocks2BlocksLocal[scol];
    mySmootherCreator.set_operator(get_operator()->get_blocked_matrix_rcp(scols, scols));
    mySmootherCreator.set_verbosity(get_verbosity());
    if (SuperBlocks2Blocks[scol].size() == 1)
    {
      int thisblock = SuperBlocks2Blocks[scol][0];
      mySmootherCreator.set_block(thisblock);
      mySmootherCreator.set_null_space(get_null_space_all_blocks()[thisblock]);
    }
    else
    {
      mySmootherCreator.set_blocks(SuperBlocks2Blocks[scol]);
      mySmootherCreator.set_null_space_all_blocks(get_null_space_all_blocks());
    }

    SubSmoothers[scol] = mySmootherCreator.create();
  }

  // =============================================================
  // Construct BGS smoother
  // =============================================================

  return Teuchos::make_rcp<BgsSmoother>(
      get_operator(), SubSmoothers, SuperBlocks2BlocksLocal, iter, omega, iters, omegas);
}



/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::BgsSmootherFactory::parse_smoother_names(
    const std::string& smoothers_string, std::vector<std::string>& smoothers_vector,
    std::vector<std::vector<int>> superblocks)
{
  if (smoothers_string == "none")
  {
    unsigned NumSuperBlocks = superblocks.size();
    smoothers_vector.resize(0);
    for (unsigned i = 0; i < NumSuperBlocks; i++)
    {
      if (0 == (superblocks[i].size()))
        FOUR_C_THROW("Something wrong related with how the blocks are set in your xml file");
      else if (1 == (superblocks[i].size()))
        smoothers_vector.emplace_back("IFPACK");
      else
        smoothers_vector.emplace_back("BGS");
    }
  }
  else
  {
    smoothers_vector.resize(0);
    std::string buf = "";
    for (char i : smoothers_string)
    {
      std::string ch(1, i);
      if (ch == ",")
      {
        smoothers_vector.push_back(buf);
        buf = "";
      }
      else
        buf += ch;
    }
    if (not(buf == "")) smoothers_vector.push_back(buf);
    buf = "";
  }

  if (smoothers_vector.size() != superblocks.size())
    FOUR_C_THROW("Not given enough subsmoothers! Fix your xml file.");
}

FOUR_C_NAMESPACE_CLOSE
