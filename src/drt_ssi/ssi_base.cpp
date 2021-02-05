/*----------------------------------------------------------------------*/
/*! \file
 \brief base class for all scalar structure algorithms

 \level 1


 *------------------------------------------------------------------------------------------------*/

#include "ssi_base.H"

#include "ssi_clonestrategy.H"
#include "ssi_coupling.H"
#include "ssi_partitioned.H"
#include "ssi_resulttest.H"
#include "ssi_str_model_evaluator_partitioned.H"
#include "ssi_utils.H"

#include "../drt_adapter/adapter_coupling.H"
#include "../drt_adapter/adapter_scatra_base_algorithm.H"
#include "../drt_adapter/ad_str_factory.H"
#include "../drt_adapter/ad_str_ssiwrapper.H"
#include "../drt_adapter/ad_str_structure_new.H"

#include "../drt_inpar/inpar_volmortar.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_utils_createdis.H"
#include "../drt_lib/drt_utils_gid_vector.H"
#include "../drt_lib/drt_utils_vector.H"

#include "../drt_scatra/scatra_timint_implicit.H"
#include "../drt_scatra_ele/scatra_ele.H"

#include "../linalg/linalg_utils_sparse_algebra_create.H"
#include "../linalg/linalg_utils_sparse_algebra_manipulation.H"

#include "../drt_lib/drt_utils_parallel.H"

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
SSI::SSIBase::SSIBase(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams)
    : AlgorithmBase(comm, globaltimeparams),
      fieldcoupling_(Teuchos::getIntegralValue<INPAR::SSI::FieldCoupling>(
          DRT::Problem::Instance()->SSIControlParams(), "FIELDCOUPLING")),
      icoup_structure_(Teuchos::null),
      icoup_structure_3_domain_intersection_(Teuchos::null),
      isinit_(false),
      issetup_(false),
      is_scatra_manifold_(
          DRT::INPUT::IntegralValue<bool>(globaltimeparams.sublist("MANIFOLD"), "ADD_MANIFOLD")),
      iter_(0),
      maps_coup_struct_(Teuchos::null),
      maps_coup_struct_3_domain_intersection_(Teuchos::null),
      map_structure_condensed_(Teuchos::null),
      map_structure_manifold_(Teuchos::null),
      meshtying_3_domain_intersection_(DRT::INPUT::IntegralValue<bool>(
          DRT::Problem::Instance()->ScalarTransportDynamicParams().sublist("S2I COUPLING"),
          "MESHTYING_3_DOMAIN_INTERSECTION")),
      scatra_base_algorithm_(Teuchos::null),
      scatra_manifold_base_algorithm_(Teuchos::null),
      slave_side_converter_(Teuchos::null),
      ssicoupling_(Teuchos::null),
      ssiinterfacemeshtying_(
          DRT::Problem::Instance()->GetDis("structure")->GetCondition("SSIInterfaceMeshtying") !=
          nullptr),
      structure_(Teuchos::null),
      struct_adapterbase_ptr_(Teuchos::null),
      temperature_funct_num_(
          DRT::Problem::Instance()->ELCHControlParams().get<int>("TEMPERATURE_FROM_FUNCT")),
      temperature_vector_(Teuchos::null),
      use_old_structure_(false),  // todo temporary flag
      zeros_structure_(Teuchos::null),
      zeros_structure_manifold_(Teuchos::null)
{
  // Keep this constructor empty!
  // First do everything on the more basic objects like the discretizations, like e.g.
  // redistribution of elements. Only then call the setup to this class. This will call the setup to
  // all classes in the inheritance hierarchy. This way, this class may also override a method that
  // is called during Setup() in a base class.
}

/*----------------------------------------------------------------------*
 | Init this class                                          rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIBase::Init(const Epetra_Comm& comm, const Teuchos::ParameterList& globaltimeparams,
    const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& structparams,
    const std::string& struct_disname, const std::string& scatra_disname, bool isAle)
{
  // reset the setup flag
  SetIsSetup(false);

  // get the global problem
  DRT::Problem* problem = DRT::Problem::Instance();

  // 3.- Create the two uncoupled subproblems.
  // access the structural discretization
  Teuchos::RCP<DRT::Discretization> structdis = DRT::Problem::Instance()->GetDis(struct_disname);

  // !!! TIME PARAMETER HANDLING !!!
  // Determine which time params to use to build the single fields.
  // In case of time stepping, time params have to be read from single field sections.
  // In case of equal timestep size for all fields the time params are controlled solely
  // by the problem section (e.g. ---SSI DYNAMIC or ---CELL DYNAMIC).
  //
  // access the ssi dynamic params
  const Teuchos::ParameterList* structtimeparams = &globaltimeparams;
  const Teuchos::ParameterList* scatratimeparams = &globaltimeparams;
  if (DRT::INPUT::IntegralValue<int>(globaltimeparams, "DIFFTIMESTEPSIZE"))
  {
    structtimeparams = &structparams;
    scatratimeparams = &scatraparams;
  }

  // do discretization specific setup (e.g. clone discr. scatra from structure)
  InitDiscretizations(comm, struct_disname, scatra_disname);

  // we do not construct a structure here, in case it
  // was built externally and handed into this object.
  if (struct_adapterbase_ptr_ == Teuchos::null)
  {
    // create structural field
    // todo FIX THIS !!!!
    // Decide whether to use old structural time integration or new structural time integration.
    // This should be removed as soon as possible!
    // Also all structural elements need to be adapted first!
    // Then, we can switch the remaining ssi tests using the old time integration to the new one,
    // i.e.: todo
    //
    // build structure
    if (structparams.get<std::string>("INT_STRATEGY") == "Standard")
    {
      struct_adapterbase_ptr_ = ADAPTER::STR::BuildStructureAlgorithm(structparams);

      // initialize structure base algorithm
      struct_adapterbase_ptr_->Init(
          *structtimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis);
    }
    else if (structparams.get<std::string>("INT_STRATEGY") ==
             "Old")  // todo this is the part that should be removed !
    {
      if (comm.MyPID() == 0)
      {
        std::cout << "\n"
                  << " USING OLD STRUCTURAL TIME INTEGRATION FOR STRUCTURE-SCATRA-INTERACTION!\n"
                     " FIX THIS! THIS IS ONLY SUPPOSED TO BE TEMPORARY!"
                     "\n"
                  << std::endl;
      }

      use_old_structure_ = true;

      Teuchos::RCP<ADAPTER::StructureBaseAlgorithm> structure =
          Teuchos::rcp(new ADAPTER::StructureBaseAlgorithm(
              *structtimeparams, const_cast<Teuchos::ParameterList&>(structparams), structdis));
      structure_ = Teuchos::rcp_dynamic_cast<::ADAPTER::SSIStructureWrapper>(
          structure->StructureField(), true);
      if (structure_ == Teuchos::null)
        dserror("cast from ADAPTER::Structure to ADAPTER::SSIStructureWrapper failed");
    }
    else
    {
      dserror(
          "Unknown time integration requested!\n"
          "Set parameter INT_STRATEGY to Standard in ---STRUCTURAL DYNAMIC section!\n"
          "If you want to use yet unsupported elements or algorithms,\n"
          "set INT_STRATEGY to Old in ---STRUCUTRAL DYNAMIC section!");
    }

  }  // if structure_ not set from outside

  // create and initialize scatra base algorithm.
  // scatra time integrator constructed and initialized inside.
  // mesh is written inside. cloning must happen before!
  scatra_base_algorithm_ = Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm());
  ScaTraBaseAlgorithm()->Init(*scatratimeparams, scatraparams,
      problem->SolverParams(scatraparams.get<int>("LINEAR_SOLVER")), scatra_disname, isAle);

  // create and initialize scatra base algorithm for manifolds
  if (IsScaTraManifold())
  {
    scatra_manifold_base_algorithm_ = Teuchos::rcp(new ADAPTER::ScaTraBaseAlgorithm());

    ScaTraManifoldBaseAlgorithm()->Init(*scatratimeparams,
        SSI::UTILS::CloneScaTraManifoldParams(
            scatraparams, globaltimeparams.sublist("MANIFOLD"), Comm()),
        problem->SolverParams(globaltimeparams.sublist("MANIFOLD").get<int>("LINEAR_SOLVER")),
        "scatra_manifold", isAle);
  }

  const RedistributionType redistribution_type =
      InitFieldCoupling(comm, struct_disname, scatra_disname);

  // is adaptive time stepping activated?
  if (DRT::INPUT::IntegralValue<bool>(globaltimeparams, "ADAPTIVE_TIMESTEPPING"))
  {
    // safety check: adaptive time stepping in one of the subproblems?
    if (!DRT::INPUT::IntegralValue<bool>(scatraparams, "ADAPTIVE_TIMESTEPPING"))
    {
      dserror(
          "Must provide adaptive time stepping algorithim in one of the subproblems. (Currently "
          "just ScTra)");
    }
    if (DRT::INPUT::IntegralValue<int>(structparams.sublist("TIMEADAPTIVITY"), "KIND") !=
        INPAR::STR::timada_kind_none)
      dserror("Adaptive time stepping in SSI currently just from ScaTra");
    if (DRT::INPUT::IntegralValue<int>(structparams, "DYNAMICTYP") == INPAR::STR::dyna_ab2)
      dserror("Currently, only one step methods are allowed for adaptive time stepping");
  }

  // set isinit_ flag true
  SetIsInit(true);

  if (redistribution_type != SSI::RedistributionType::none) Redistribute(redistribution_type);
}

/*----------------------------------------------------------------------*
 | Setup this class                                         rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIBase::Setup()
{
  // check initialization
  CheckIsInit();

  // set up helper class for field coupling
  ssicoupling_->Setup();

  // set up scalar transport field
  ScaTraField()->Setup();
  if (IsScaTraManifold()) ScaTraManifold()->Setup();

  // only relevant for new structural time integration
  // only if adapter base has not already been set up outside
  if (not use_old_structure_ and not struct_adapterbase_ptr_->IsSetup())
  {
    // set up structural model evaluator
    SetupModelEvaluator();

    // pass initial scalar field to structural discretization to correctly compute initial
    // accelerations
    if (Teuchos::getIntegralValue<INPAR::SSI::SolutionSchemeOverFields>(
            DRT::Problem::Instance()->SSIControlParams(), "COUPALGO") !=
        INPAR::SSI::SolutionSchemeOverFields::ssi_OneWay_SolidToScatra)
      ssicoupling_->SetScalarField(
          *DRT::Problem::Instance()->GetDis("structure"), ScaTraField()->Phinp());

    // temperature is non primary variable. Only set, if function for temperature is given
    if (temperature_funct_num_ != -1)
    {
      temperature_vector_ = Teuchos::rcp(
          new Epetra_Vector(*DRT::Problem::Instance()->GetDis("structure")->DofRowMap(2), true));

      temperature_vector_->PutScalar(
          DRT::Problem::Instance()->Funct(temperature_funct_num_ - 1).EvaluateTime(Time()));

      ssicoupling_->SetTemperatureField(
          *DRT::Problem::Instance()->GetDis("structure"), temperature_vector_);
    }

    // set up structural base algorithm
    struct_adapterbase_ptr_->Setup();

    // get wrapper and cast it to specific type
    // do not do so, in case the wrapper has already been set from outside
    if (structure_ == Teuchos::null)
      structure_ = Teuchos::rcp_dynamic_cast<::ADAPTER::SSIStructureWrapper>(
          struct_adapterbase_ptr_->StructureField());

    if (structure_ == Teuchos::null)
    {
      dserror(
          "No valid pointer to ADAPTER::SSIStructureWrapper !\n"
          "Either cast failed, or no valid wrapper was set using SetStructureWrapper(...) !");
    }
  }

  // for old structural time integration
  else if (use_old_structure_)
    structure_->Setup();

  // check maps from scalar transport and structure discretizations
  if (ScaTraField()->DofRowMap()->NumGlobalElements() == 0)
    dserror("Scalar transport discretization does not have any degrees of freedom!");
  if (structure_->DofRowMap()->NumGlobalElements() == 0)
    dserror("Structure discretization does not have any degrees of freedom!");

  // set up materials
  ssicoupling_->AssignMaterialPointers(
      structure_->Discretization(), ScaTraField()->Discretization());

  // set up scatra-scatra interface coupling
  if (SSIInterfaceMeshtying())
  {
    // check for consistent parameterization of these conditions
    Teuchos::RCP<DRT::Discretization> structdis = DRT::Problem::Instance()->GetDis("structure");
    // get ssi to be tested
    std::vector<DRT::Condition*> ssiconditions;
    structdis->GetCondition("SSIInterfaceMeshtying", ssiconditions);
    SSI::UTILS::CheckConsistencyWithS2IMeshtyingCondition(ssiconditions, structdis);

    // set up scatra-scatra interface coupling adapter for structure field
    icoup_structure_ = SSI::UTILS::SetupInterfaceCouplingAdapterStructure(structdis,
        Meshtying3DomainIntersection(), "SSIInterfaceMeshtying", "SSIMeshtying3DomainIntersection");

    if (Meshtying3DomainIntersection())
    {
      icoup_structure_3_domain_intersection_ =
          SSI::UTILS::SetupInterfaceCouplingAdapterStructure3DomainIntersection(
              structdis, "SSIMeshtying3DomainIntersection");

      auto map1 =
          Teuchos::rcp(new const Epetra_Map(*InterfaceCouplingAdapterStructure()->SlaveDofMap()));
      auto map2 = Teuchos::rcp(new const Epetra_Map(
          *InterfaceCouplingAdapterStructure3DomainIntersection()->SlaveDofMap()));
      auto map3 = LINALG::MultiMapExtractor::MergeMaps({map1, map2});

      // set up map for interior and master-side structural degrees of freedom
      map_structure_condensed_ =
          LINALG::SplitMap(*structure_->Discretization()->DofRowMap(), *map3);
    }
    else
    {
      // set up map for interior and master-side structural degrees of freedom
      map_structure_condensed_ = LINALG::SplitMap(*structure_->Discretization()->DofRowMap(),
          *InterfaceCouplingAdapterStructure()->SlaveDofMap());
    }


    slave_side_converter_ = Teuchos::rcp(new SSI::UTILS::SSISlaveSideConverter(
        icoup_structure_, icoup_structure_3_domain_intersection_, Meshtying3DomainIntersection()));

    // set up structural map extractor holding interior and interface maps of degrees of freedom
    std::vector<Teuchos::RCP<const Epetra_Map>> maps_surf(0, Teuchos::null);
    maps_surf.emplace_back(LINALG::SplitMap(
        *map_structure_condensed_, *InterfaceCouplingAdapterStructure()->MasterDofMap()));
    maps_surf.emplace_back(InterfaceCouplingAdapterStructure()->SlaveDofMap());
    maps_surf.emplace_back(InterfaceCouplingAdapterStructure()->MasterDofMap());
    maps_coup_struct_ = Teuchos::rcp(
        new LINALG::MultiMapExtractor(*structure_->Discretization()->DofRowMap(), maps_surf));
    maps_coup_struct_->CheckForValidMapExtractor();

    if (Meshtying3DomainIntersection())
    {
      std::vector<Teuchos::RCP<const Epetra_Map>> maps_line(0, Teuchos::null);
      maps_line.emplace_back(
          LINALG::SplitMap(*InterfaceCouplingAdapterStructure3DomainIntersection()->MasterDofMap(),
              *InterfaceCouplingAdapterStructure3DomainIntersection()->MasterDofMap()));
      maps_line.emplace_back(InterfaceCouplingAdapterStructure3DomainIntersection()->SlaveDofMap());
      maps_line.emplace_back(
          InterfaceCouplingAdapterStructure3DomainIntersection()->MasterDofMap());
      maps_coup_struct_3_domain_intersection_ = Teuchos::rcp(
          new LINALG::MultiMapExtractor(*structure_->Discretization()->DofRowMap(), maps_line));
      maps_coup_struct_3_domain_intersection_->CheckForValidMapExtractor();
    }
  }

  // create map of dofs on structure discretization that have the same nodes as manifold (scatra)
  if (IsScaTraManifold())
  {
    std::vector<DRT::Condition*> conditions(0, nullptr);
    StructureField()->Discretization()->GetCondition("SSISurfaceManifold", conditions);
    if (conditions.empty()) dserror("Condition SSISurfaceManifold not found");

    // Build GID vector of nodes on this proc, sort, and remove duplicates
    std::vector<int> condition_node_vec;
    for (auto* condition : conditions)
    {
      DRT::UTILS::AddOwnedNodeGIDVector(
          StructureField()->Discretization(), *condition->Nodes(), condition_node_vec);
    }
    DRT::UTILS::SortAndRemoveDuplicateVectorElements(condition_node_vec);

    // Build GID vector of dofs
    std::vector<int> condition_dof_vec;
    for (int condition_node : condition_node_vec)
    {
      const int dim = StructureField()->Discretization()->gNode(condition_node)->Dim();
      for (int j = 0; j < dim; ++j) condition_dof_vec.emplace_back(condition_node * dim + j);
    }

    // maps of conditioned dofs and other dofs
    const auto condition_dof_map = Teuchos::rcp(
        new const Epetra_Map(-1, condition_dof_vec.size(), &condition_dof_vec[0], 0, Comm()));
    const auto non_condition_dof_map =
        LINALG::SplitMap(*StructureField()->Discretization()->DofRowMap(), *condition_dof_map);

    map_structure_manifold_ = Teuchos::rcp(new LINALG::MapExtractor(
        *StructureField()->DofRowMap(), non_condition_dof_map, condition_dof_map));
  }

  // construct vector of zeroes
  zeros_structure_ = LINALG::CreateVector(*structure_->DofRowMap());
  zeros_structure_manifold_ =
      IsScaTraManifold() ? LINALG::CreateVector(*map_structure_manifold_->Map(0)) : Teuchos::null;

  // set flag
  SetIsSetup(true);
}

/*----------------------------------------------------------------------*
 | Setup the discretizations                                rauch 08/16 |
 *----------------------------------------------------------------------*/
void SSI::SSIBase::InitDiscretizations(
    const Epetra_Comm& comm, const std::string& struct_disname, const std::string& scatra_disname)
{
  // Scheme   : the structure discretization is received from the input. Then, an ale-scatra disc.
  // is cloned.

  DRT::Problem* problem = DRT::Problem::Instance();

  auto structdis = problem->GetDis(struct_disname);
  auto scatradis = problem->GetDis(scatra_disname);

  if (scatradis->NumGlobalNodes() == 0)
  {
    if (fieldcoupling_ != INPAR::SSI::FieldCoupling::volume_match and
        fieldcoupling_ != INPAR::SSI::FieldCoupling::volumeboundary_match)
    {
      dserror(
          "If 'FIELDCOUPLING' is NOT 'volume_matching' or 'volumeboundary_matching' in the SSI "
          "CONTROL section cloning of the scatra discretization from the structure discretization "
          "is not supported!");
    }

    // fill scatra discretization by cloning structure discretization
    DRT::UTILS::CloneDiscretization<SSI::ScatraStructureCloneStrategy>(structdis, scatradis);
    scatradis->FillComplete();

    // create discretization for scatra manifold based on SSISurfaceManifold condition
    if (IsScaTraManifold())
    {
      auto scatra_manifold_dis = problem->GetDis("scatra_manifold");
      DRT::UTILS::CloneDiscretizationFromCondition<SSI::ScatraStructureCloneStrategyManifold>(
          *structdis, *scatra_manifold_dis, "SSISurfaceManifold");
      {
        std::map<std::string, std::string> conditions_to_copy;
        conditions_to_copy.insert(
            std::pair<std::string, std::string>("SSISurfaceManifold", "ScatraPartitioning"));
        DRT::UTILS::DiscretizationCreatorBase creator;
        creator.CopyConditions(*structdis, *scatra_manifold_dis, conditions_to_copy);
      }
      scatra_manifold_dis->FillComplete();
    }
  }
  else
  {
    if (fieldcoupling_ == INPAR::SSI::FieldCoupling::volume_match)
    {
      dserror(
          "Reading a TRANSPORT discretization from the .dat file for the input parameter "
          "'FIELDCOUPLING volume_matching' in the SSI CONTROL section is not supported! As this "
          "coupling relies on matching node (and sometimes element) IDs, the ScaTra discretization "
          "is cloned from the structure discretization. Delete the ScaTra discretization from your "
          "input file.");
    }

    // copy conditions
    // this is actually only needed for copying TRANSPORT DIRICHLET/NEUMANN CONDITIONS
    // as standard DIRICHLET/NEUMANN CONDITIONS
    std::map<std::string, std::string> conditions_to_copy;
    SSI::ScatraStructureCloneStrategy clonestrategy;
    conditions_to_copy = clonestrategy.ConditionsToCopy();
    DRT::UTILS::DiscretizationCreatorBase creator;
    creator.CopyConditions(*scatradis, *scatradis, conditions_to_copy);

    // safety check, since it is not reasonable to have SOLIDSCATRA or SOLIDPOROP1 Elements with a
    // SCATRA::ImplType != 'impltype_undefined' if they are not cloned! Therefore loop over all
    // structure elements and check the impltype
    for (int i = 0; i < structdis->NumMyColElements(); ++i)
    {
      if (clonestrategy.GetImplType(structdis->lColElement(i)) != INPAR::SCATRA::impltype_undefined)
      {
        dserror(
            "A TRANSPORT discretization is read from the .dat file, which is fine since the scatra "
            "discretization is not cloned from "
            "the structure discretization. But in the STRUCTURE ELEMENTS section of the .dat file "
            "an ImplType that is NOT 'Undefined' is prescribed "
            "which does not make sense if you don't want to clone the structure discretization. "
            "Change the ImplType to 'Undefined' "
            "or decide to clone the scatra discretization from the structure discretization.");
      }
    }
  }
}

/*----------------------------------------------------------------------*
 | Setup ssi coupling object                                rauch 08/16 |
 *----------------------------------------------------------------------*/
SSI::RedistributionType SSI::SSIBase::InitFieldCoupling(
    const Epetra_Comm& comm, const std::string& struct_disname, const std::string& scatra_disname)
{
  // initialize return variable
  RedistributionType redistribution_required = SSI::RedistributionType::none;

  DRT::Problem* problem = DRT::Problem::Instance();
  auto structdis = problem->GetDis(struct_disname);
  auto scatradis = problem->GetDis(scatra_disname);
  Teuchos::RCP<DRT::Discretization> scatra_manifold_dis(Teuchos::null);
  if (IsScaTraManifold()) scatra_manifold_dis = problem->GetDis("scatra_manifold");

  // safety check
  {
    // check for ssi coupling condition
    std::vector<DRT::Condition*> ssicoupling;
    scatradis->GetCondition("SSICoupling", ssicoupling);
    const bool havessicoupling = (ssicoupling.size() > 0);

    if (havessicoupling and (fieldcoupling_ != INPAR::SSI::FieldCoupling::boundary_nonmatch and
                                fieldcoupling_ != INPAR::SSI::FieldCoupling::volumeboundary_match))
    {
      dserror(
          "SSICoupling condition only valid in combination with FIELDCOUPLING set to "
          "'boundary_nonmatching' or 'volumeboundary_matching' in SSI DYNAMIC section. ");
    }

    if (fieldcoupling_ == INPAR::SSI::FieldCoupling::volume_nonmatch)
    {
      const Teuchos::ParameterList& volmortarparams = DRT::Problem::Instance()->VolmortarParams();
      if (DRT::INPUT::IntegralValue<INPAR::VOLMORTAR::CouplingType>(
              volmortarparams, "COUPLINGTYPE") != INPAR::VOLMORTAR::couplingtype_coninter)
      {
        dserror(
            "Volmortar coupling only tested for consistent interpolation, "
            "i.e. 'COUPLINGTYPE consint' in VOLMORTAR COUPLING section. Try other couplings at own "
            "risk.");
      }
    }
    if (IsScaTraManifold() and fieldcoupling_ != INPAR::SSI::FieldCoupling::volumeboundary_match)
      dserror("Solving manifolds only in combination with matching volumes and boundaries");
  }

  // build SSI coupling class
  switch (fieldcoupling_)
  {
    case INPAR::SSI::FieldCoupling::volume_match:
      ssicoupling_ = Teuchos::rcp(new SSICouplingMatchingVolume());
      break;
    case INPAR::SSI::FieldCoupling::volume_nonmatch:
      ssicoupling_ = Teuchos::rcp(new SSICouplingNonMatchingVolume());
      // redistribution is still performed inside
      redistribution_required = SSI::RedistributionType::binning;
      break;
    case INPAR::SSI::FieldCoupling::boundary_nonmatch:
      ssicoupling_ = Teuchos::rcp(new SSICouplingNonMatchingBoundary());
      break;
    case INPAR::SSI::FieldCoupling::volumeboundary_match:
      ssicoupling_ = Teuchos::rcp(new SSICouplingMatchingVolumeAndBoundary());
      redistribution_required = SSI::RedistributionType::match;
      break;
    default:
      dserror("unknown type of field coupling for SSI!");
      break;
  }

  // initialize coupling objects including dof sets
  ssicoupling_->Init(problem->NDim(), structdis, scatradis, scatra_manifold_dis);

  return redistribution_required;
}

/*----------------------------------------------------------------------*
 | read restart information for given time step (public)   vuong 01/12  |
 *----------------------------------------------------------------------*/
void SSI::SSIBase::ReadRestart(int restart)
{
  if (restart)
  {
    structure_->ReadRestart(restart);

    const Teuchos::ParameterList& ssidyn = DRT::Problem::Instance()->SSIControlParams();
    const bool restartfromstructure =
        DRT::INPUT::IntegralValue<int>(ssidyn, "RESTART_FROM_STRUCTURE");

    if (not restartfromstructure)  // standard restart
    {
      ScaTraField()->ReadRestart(restart);
      if (IsScaTraManifold()) ScaTraManifold()->ReadRestart(restart);
    }
    else  // restart from structure simulation
    {
      // Since there is no restart output for the scatra fiels available, we only have to fix the
      // time and step counter
      ScaTraField()->SetTimeStep(structure_->TimeOld(), restart);
      if (IsScaTraManifold()) ScaTraManifold()->SetTimeStep(structure_->TimeOld(), restart);
    }

    SetTimeStep(structure_->TimeOld(), restart);
  }

  // Material pointers to other field were deleted during ReadRestart().
  // They need to be reset.
  ssicoupling_->AssignMaterialPointers(
      structure_->Discretization(), ScaTraField()->Discretization());
}

/*----------------------------------------------------------------------*
 | read restart information for given time (public)        AN, JH 10/14 |
 *----------------------------------------------------------------------*/
void SSI::SSIBase::ReadRestartfromTime(double restarttime)
{
  if (restarttime > 0.0)
  {
    const int restartstructure = SSI::UTILS::CheckTimeStepping(structure_->Dt(), restarttime);
    const int restartscatra = SSI::UTILS::CheckTimeStepping(ScaTraField()->Dt(), restarttime);
    const int restartscatramanifold =
        IsScaTraManifold() ? SSI::UTILS::CheckTimeStepping(ScaTraManifold()->Dt(), restarttime)
                           : -1;

    structure_->ReadRestart(restartstructure);

    const Teuchos::ParameterList& ssidyn = DRT::Problem::Instance()->SSIControlParams();
    const bool restartfromstructure =
        DRT::INPUT::IntegralValue<int>(ssidyn, "RESTART_FROM_STRUCTURE");

    if (not restartfromstructure)  // standard restart
    {
      ScaTraField()->ReadRestart(restartscatra);
      if (IsScaTraManifold()) ScaTraManifold()->ReadRestart(restartscatramanifold);
    }
    else  // restart from structure simulation
    {
      // Since there is no restart output for the scatra fiels available, we only have to fix the
      // time and step counter
      ScaTraField()->SetTimeStep(structure_->TimeOld(), restartscatra);
      if (IsScaTraManifold())
        ScaTraManifold()->SetTimeStep(structure_->TimeOld(), restartscatramanifold);
    }

    SetTimeStep(structure_->TimeOld(), restartstructure);
  }

  // Material pointers to other field were deleted during ReadRestart().
  // They need to be reset.
  ssicoupling_->AssignMaterialPointers(
      structure_->Discretization(), ScaTraField()->Discretization());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::TestResults(const Epetra_Comm& comm) const
{
  DRT::Problem* problem = DRT::Problem::Instance();

  problem->AddFieldTest(structure_->CreateFieldTest());
  problem->AddFieldTest(ScaTraBaseAlgorithm()->CreateScaTraFieldTest());
  if (IsScaTraManifold())
    problem->AddFieldTest(ScaTraManifoldBaseAlgorithm()->CreateScaTraFieldTest());
  problem->AddFieldTest(Teuchos::rcp(new SSI::SSIResultTest(Teuchos::rcp(this, false))));
  problem->TestAll(comm);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::SetStructSolution(
    Teuchos::RCP<const Epetra_Vector> disp, Teuchos::RCP<const Epetra_Vector> vel)
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  SetMeshDisp(disp);
  SetVelocityFields(vel);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::SetScatraSolution(Teuchos::RCP<const Epetra_Vector> phi)
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  ssicoupling_->SetScalarField(*structure_->Discretization(), phi);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::EvaluateAndSetTemperatureField()
{
  // temperature is non primary variable. Only set, if function for temperature is given
  if (temperature_funct_num_ != -1)
  {
    // evaluate temperature at current time and put to scalar
    const double temperature =
        DRT::Problem::Instance()->Funct(temperature_funct_num_ - 1).EvaluateTime(Time());
    temperature_vector_->PutScalar(temperature);

    // set temperature vector to structure discretization
    ssicoupling_->SetTemperatureField(*structure_->Discretization(), temperature_vector_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::SetVelocityFields(Teuchos::RCP<const Epetra_Vector> vel)
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  ssicoupling_->SetVelocityFields(ScaTraBaseAlgorithm(), zeros_structure_, vel);
  if (IsScaTraManifold())
  {
    ssicoupling_->SetVelocityFields(ScaTraManifoldBaseAlgorithm(), zeros_structure_manifold_,
        MapStructureManifold()->ExtractVector(vel, 0));
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::SetMeshDisp(Teuchos::RCP<const Epetra_Vector> disp)
{
  // safety checks
  CheckIsInit();
  CheckIsSetup();

  ssicoupling_->SetMeshDisp(ScaTraBaseAlgorithm(), disp);
  if (IsScaTraManifold())
  {
    ssicoupling_->SetMeshDisp(
        ScaTraManifoldBaseAlgorithm(), MapStructureManifold()->ExtractVector(disp, 0));
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::SetDtFromScaTraToStructure()
{
  // change current time and time step of structure according to ScaTra
  StructureField()->SetDt(ScaTraField()->Dt());
  StructureField()->SetTimen(ScaTraField()->Time());
  StructureField()->PostUpdate();

  // change current time and time step of this algorithm according to ScaTra
  SetTimeStep(ScaTraField()->Time(), Step());
  SetDt(ScaTraField()->Dt());
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void SSI::SSIBase::Redistribute(const RedistributionType redistribution_type)
{
  DRT::Problem* problem = DRT::Problem::Instance();

  auto structdis = problem->GetDis("structure");
  auto scatradis = problem->GetDis("scatra");
  if (redistribution_type == SSI::RedistributionType::match)
  {
    if (IsScaTraManifold())
    {
      std::vector<Teuchos::RCP<DRT::Discretization>> dis;
      dis.push_back(structdis);
      DRT::UTILS::RedistributeDiscretizationsByBinning(dis, false);

      DRT::UTILS::MatchElementDistributionOfMatchingConditionedElements(*structdis,
          *problem->GetDis("scatra_manifold"), "SSISurfaceManifold", "SSISurfaceManifold");

      DRT::UTILS::MatchElementDistributionOfMatchingDiscretizations(*structdis, *scatradis);

      structdis->ExportColumnNodes(*scatradis->NodeColMap());
    }
    else
    {
      // first we bin the scatra discretization
      std::vector<Teuchos::RCP<DRT::Discretization>> dis;
      dis.push_back(scatradis);
      DRT::UTILS::RedistributeDiscretizationsByBinning(dis, false);

      DRT::UTILS::MatchElementDistributionOfMatchingConditionedElements(
          *scatradis, *scatradis, "ScatraHeteroReactionMaster", "ScatraHeteroReactionSlave");

      // now we redistribute the structure dis to match the scatra dis
      DRT::UTILS::MatchElementDistributionOfMatchingDiscretizations(*scatradis, *structdis);
    }
  }
  else if (redistribution_type == SSI::RedistributionType::binning)
  {
    // create vector of discr.
    std::vector<Teuchos::RCP<DRT::Discretization>> dis;
    dis.push_back(structdis);
    dis.push_back(scatradis);

    DRT::UTILS::RedistributeDiscretizationsByBinning(dis, false);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> SSI::SSIBase::ScaTraField() const
{
  return scatra_base_algorithm_->ScaTraField();
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> SSI::SSIBase::ScaTraManifold() const
{
  return scatra_manifold_base_algorithm_->ScaTraField();
}
