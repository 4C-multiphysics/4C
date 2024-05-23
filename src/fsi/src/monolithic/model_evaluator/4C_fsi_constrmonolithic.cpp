/*----------------------------------------------------------------------------*/
/*! \file


\brief Solve FSI problem with volume constraints

\level 2
*/

/*----------------------------------------------------------------------------*/

#include "4C_fsi_constrmonolithic.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_fluid_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_manager.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_overlapprec_fsiamg.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_io_control.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_structure_aux.hpp"

#include <NOX_Epetra_LinearSystem.H>
#include <NOX_Epetra_LinearSystem_AztecOO.H>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::ConstrMonolithic::ConstrMonolithic(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : BlockMonolithic(comm, timeparams), conman_(StructureField()->get_constraint_manager())
{
  icoupfa_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());
  coupsaout_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());
  coupfsout_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());
  coupfaout_ = Teuchos::rcp(new CORE::ADAPTER::Coupling());

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithic::GeneralSetup()
{
  const Teuchos::ParameterList& fsidyn = GLOBAL::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  linearsolverstrategy_ =
      CORE::UTILS::IntegralValue<INPAR::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

  set_default_parameters(fsidyn, NOXParameterList());

  // ToDo: Set more detailed convergence tolerances like in standard FSI
  // additionally set tolerance for volume constraint
  NOXParameterList().set("Norm abs vol constr", fsimono.get<double>("CONVTOL"));

  //-----------------------------------------------------------------------------
  // ordinary fsi coupling
  //-----------------------------------------------------------------------------

  // right now we use matching meshes at the interface

  CORE::ADAPTER::Coupling& coupsf = structure_fluid_coupling();
  CORE::ADAPTER::Coupling& coupsa = structure_ale_coupling();
  CORE::ADAPTER::Coupling& coupfa = FluidAleCoupling();

  // structure to fluid
  const int ndim = GLOBAL::Problem::Instance()->NDim();
  coupsf.setup_condition_coupling(*StructureField()->Discretization(),
      StructureField()->Interface()->FSICondMap(), *FluidField()->Discretization(),
      FluidField()->Interface()->FSICondMap(), "FSICoupling", ndim);

  // structure to ale

  coupsa.setup_condition_coupling(*StructureField()->Discretization(),
      StructureField()->Interface()->FSICondMap(), *AleField()->Discretization(),
      AleField()->Interface()->FSICondMap(), "FSICoupling", ndim);

  // fluid to ale at the interface

  icoupfa_->setup_condition_coupling(*FluidField()->Discretization(),
      FluidField()->Interface()->FSICondMap(), *AleField()->Discretization(),
      AleField()->Interface()->FSICondMap(), "FSICoupling", ndim);

  // In the following we assume that both couplings find the same dof
  // map at the structural side. This enables us to use just one
  // interface dof map for all fields and have just one transfer
  // operator from the interface map to the full field map.
  if (not coupsf.MasterDofMap()->SameAs(*coupsa.MasterDofMap()))
    FOUR_C_THROW("structure interface dof maps do not match");

  if (coupsf.MasterDofMap()->NumGlobalElements() == 0)
    FOUR_C_THROW("No nodes in matching FSI interface. Empty FSI coupling condition?");

  // the fluid-ale coupling always matches
  const Epetra_Map* fluidnodemap = FluidField()->Discretization()->NodeRowMap();
  const Epetra_Map* alenodemap = AleField()->Discretization()->NodeRowMap();

  coupfa.SetupCoupling(*FluidField()->Discretization(), *AleField()->Discretization(),
      *fluidnodemap, *alenodemap, ndim);

  FluidField()->SetMeshMap(coupfa.MasterDofMap());

  aleresidual_ = Teuchos::rcp(new Epetra_Vector(*AleField()->Interface()->Map(0)));

  // ---------------------------------------------------------------------------
  // Build the global Dirichlet map extractor
  //
  // Dirichlet maps for structure and fluid do not intersect with interface map.
  // ALE Dirichlet map might intersect with interface map, but ALE interface DOFs
  // are not part of the final system of equations. Hence, we just need the
  // intersection of inner ALE DOFs with Dirichlet ALE DOFs.
  std::vector<Teuchos::RCP<const Epetra_Map>> aleintersectionmaps;
  aleintersectionmaps.push_back(AleField()->GetDBCMapExtractor()->CondMap());
  aleintersectionmaps.push_back(AleField()->Interface()->OtherMap());
  Teuchos::RCP<Epetra_Map> aleintersectionmap =
      CORE::LINALG::MultiMapExtractor::IntersectMaps(aleintersectionmaps);

  // Merge Dirichlet maps of structure, fluid and ALE to global FSI Dirichlet map
  std::vector<Teuchos::RCP<const Epetra_Map>> dbcmaps;
  dbcmaps.push_back(StructureField()->GetDBCMapExtractor()->CondMap());
  dbcmaps.push_back(FluidField()->GetDBCMapExtractor()->CondMap());
  dbcmaps.push_back(aleintersectionmap);
  Teuchos::RCP<const Epetra_Map> dbcmap = CORE::LINALG::MultiMapExtractor::MergeMaps(dbcmaps);

  // Finally, create the global FSI Dirichlet map extractor
  dbcmaps_ = Teuchos::rcp(new CORE::LINALG::MapExtractor(*DofRowMap(), dbcmap, true));
  if (dbcmaps_ == Teuchos::null)
  {
    FOUR_C_THROW("Creation of FSI Dirichlet map extractor failed.");
  }
  // ---------------------------------------------------------------------------

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithic::Evaluate(Teuchos::RCP<const Epetra_Vector> step_increment)
{
  //-----------------------------------------------------------------------------
  // Increment lagrange multiplier
  //-----------------------------------------------------------------------------
  if (step_increment != Teuchos::null)
  {
    Teuchos::RCP<Epetra_Vector> lagrincr = Extractor().ExtractVector(step_increment, 3);
    conman_->UpdateTotLagrMult(lagrincr);
  }

  //-----------------------------------------------------------------------------
  // evaluation of all fields; constraints are evaluated by strucuture
  //-----------------------------------------------------------------------------
  FSI::Monolithic::Evaluate(step_increment);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithic::ScaleSystem(CORE::LINALG::BlockSparseMatrixBase& mat, Epetra_Vector& b)
{
  // should we scale the system?
  const Teuchos::ParameterList& fsidyn = GLOBAL::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = (bool)CORE::UTILS::IntegralValue<int>(fsimono, "INFNORMSCALING");

  if (scaling_infnorm)
  {
    // The matrices are modified here. Do we have to change them back later on?

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.Matrix(0, 0).EpetraMatrix();
    srowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    scolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*srowsum_);
    A->InvColSums(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.Matrix(0, 1).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0, 2).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0, 3).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(1, 0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(2, 0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(3, 0).EpetraMatrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.Matrix(2, 2).EpetraMatrix();
    arowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    acolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*arowsum_);
    A->InvColSums(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.Matrix(2, 0).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(2, 1).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(0, 2).EpetraMatrix()->RightScale(*acolsum_) or
        mat.Matrix(1, 2).EpetraMatrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");

    Teuchos::RCP<Epetra_Vector> sx = Extractor().ExtractVector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = Extractor().ExtractVector(b, 2);

    if (sx->Multiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->Multiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    Extractor().InsertVector(*sx, 0, b);
    Extractor().InsertVector(*ax, 2, b);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithic::UnscaleSolution(
    CORE::LINALG::BlockSparseMatrixBase& mat, Epetra_Vector& x, Epetra_Vector& b)
{
  const Teuchos::ParameterList& fsidyn = GLOBAL::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = (bool)CORE::UTILS::IntegralValue<int>(fsimono, "INFNORMSCALING");

  if (scaling_infnorm)
  {
    Teuchos::RCP<Epetra_Vector> sy = Extractor().ExtractVector(x, 0);
    Teuchos::RCP<Epetra_Vector> ay = Extractor().ExtractVector(x, 2);

    if (sy->Multiply(1.0, *scolsum_, *sy, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ay->Multiply(1.0, *acolsum_, *ay, 0.0)) FOUR_C_THROW("ale scaling failed");

    Extractor().InsertVector(*sy, 0, x);
    Extractor().InsertVector(*ay, 2, x);

    Teuchos::RCP<Epetra_Vector> sx = Extractor().ExtractVector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = Extractor().ExtractVector(b, 2);

    if (sx->ReciprocalMultiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->ReciprocalMultiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    Extractor().InsertVector(*sx, 0, b);
    Extractor().InsertVector(*ax, 2, b);

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.Matrix(0, 0).EpetraMatrix();
    srowsum_->Reciprocal(*srowsum_);
    scolsum_->Reciprocal(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.Matrix(0, 1).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0, 2).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(0, 3).EpetraMatrix()->LeftScale(*srowsum_) or
        mat.Matrix(1, 0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(2, 0).EpetraMatrix()->RightScale(*scolsum_) or
        mat.Matrix(3, 0).EpetraMatrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.Matrix(2, 2).EpetraMatrix();
    arowsum_->Reciprocal(*arowsum_);
    acolsum_->Reciprocal(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.Matrix(2, 0).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(2, 1).EpetraMatrix()->LeftScale(*arowsum_) or
        mat.Matrix(0, 2).EpetraMatrix()->RightScale(*acolsum_) or
        mat.Matrix(1, 2).EpetraMatrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");
  }

  // very simple hack just to see the linear solution

  Epetra_Vector r(b.Map());
  mat.Apply(x, r);
  r.Update(1., b, 1.);

  Teuchos::RCP<Epetra_Vector> sr = Extractor().ExtractVector(r, 0);
  Teuchos::RCP<Epetra_Vector> fr = Extractor().ExtractVector(r, 1);
  Teuchos::RCP<Epetra_Vector> ar = Extractor().ExtractVector(r, 2);

  // increment additional ale residual
  aleresidual_->Update(-1., *ar, 0.);

  std::ios_base::fmtflags flags = Utils()->out().flags();

  double n, ns, nf, na;
  r.Norm2(&n);
  sr->Norm2(&ns);
  fr->Norm2(&nf);
  ar->Norm2(&na);
  Utils()->out() << std::scientific << "\nlinear solver quality:\n"
                 << "L_2-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "\n";
  r.NormInf(&n);
  sr->NormInf(&ns);
  fr->NormInf(&nf);
  ar->NormInf(&na);
  Utils()->out() << "L_inf-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "\n";

  Utils()->out().flags(flags);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::LinearSystem> FSI::ConstrMonolithic::CreateLinearSystem(
    Teuchos::ParameterList& nlParams, ::NOX::Epetra::Vector& noxSoln,
    Teuchos::RCP<::NOX::Utils> utils)
{
  Teuchos::RCP<::NOX::Epetra::LinearSystem> linSys;

  Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
  Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");
  Teuchos::ParameterList& newtonParams = dirParams.sublist("Newton");
  Teuchos::ParameterList* lsParams = nullptr;

  // in case of nonlinCG the linear solver list is somewhere else
  if (dirParams.get("Method", "User Defined") == "User Defined")
    lsParams = &(newtonParams.sublist("Linear Solver"));
  else if (dirParams.get("Method", "User Defined") == "NonlinearCG")
    lsParams = &(dirParams.sublist("Nonlinear CG").sublist("Linear Solver"));
  else
    FOUR_C_THROW("Unknown nonlinear method");

  ::NOX::Epetra::Interface::Jacobian* iJac = this;
  ::NOX::Epetra::Interface::Preconditioner* iPrec = this;
  const Teuchos::RCP<Epetra_Operator> J = systemmatrix_;
  const Teuchos::RCP<Epetra_Operator> M = systemmatrix_;

  switch (linearsolverstrategy_)
  {
    case INPAR::FSI::PreconditionedKrylov:
      linSys = Teuchos::rcp(new ::NOX::Epetra::LinearSystemAztecOO(printParams, *lsParams,
          Teuchos::rcp(iJac, false), J, Teuchos::rcp(iPrec, false), M, noxSoln));
      break;
    default:
      FOUR_C_THROW("Unsupported type of monolithic solver/preconditioner!");
      break;
  }

  return linSys;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::ConstrMonolithic::CreateStatusTest(
    Teuchos::ParameterList& nlParams, Teuchos::RCP<::NOX::Epetra::Group> grp)
{
  // Create the convergence tests
  Teuchos::RCP<::NOX::StatusTest::Combo> combo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));
  Teuchos::RCP<::NOX::StatusTest::Combo> converged =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::AND));

  Teuchos::RCP<::NOX::StatusTest::MaxIters> maxiters =
      Teuchos::rcp(new ::NOX::StatusTest::MaxIters(nlParams.get("Max Iterations", 100)));
  Teuchos::RCP<::NOX::StatusTest::FiniteValue> fv =
      Teuchos::rcp(new ::NOX::StatusTest::FiniteValue);

  combo->addStatusTest(fv);
  combo->addStatusTest(converged);
  combo->addStatusTest(maxiters);

  // require one solve
  converged->addStatusTest(Teuchos::rcp(new NOX::FSI::MinIters(1)));

  // setup tests for structural displacements

  Teuchos::RCP<::NOX::StatusTest::Combo> structcombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> structureDisp = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "displacement", Extractor(), 0, nlParams.get<double>("Norm abs disp"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> structureDispUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("displacement update", Extractor(), 0,
          nlParams.get<double>("Norm abs disp"), NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(structureDisp);
  structcombo->addStatusTest(structureDisp);
  // structcombo->addStatusTest(structureDispUpdate);

  converged->addStatusTest(structcombo);

  //  // setup tests for interface
  //
  //  std::vector<Teuchos::RCP<const Epetra_Map> > interface;
  //  interface.push_back(FluidField()->Interface()->FSICondMap());
  //  interface.push_back(Teuchos::null);
  //  CORE::LINALG::MultiMapExtractor interfaceextract(*DofRowMap(),interface);
  //
  //  Teuchos::RCP<::NOX::StatusTest::Combo> interfacecombo =
  //    Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));
  //
  //  Teuchos::RCP<NOX::FSI::PartialNormF> interfaceTest =
  //    Teuchos::rcp(new NOX::FSI::PartialNormF("interface",
  //                                            interfaceextract,0,
  //                                            nlParams.get("Norm abs vel", 1.0e-6),
  //                                            ::NOX::Abstract::Vector::TwoNorm,
  //                                            NOX::FSI::PartialNormF::Scaled));
  //  Teuchos::RCP<NOX::FSI::PartialNormUpdate> interfaceTestUpdate =
  //    Teuchos::rcp(new NOX::FSI::PartialNormUpdate("interface update",
  //                                                 interfaceextract,0,
  //                                                 nlParams.get("Norm abs vel", 1.0e-6),
  //                                                 NOX::FSI::PartialNormUpdate::Scaled));
  //
  //  AddStatusTest(interfaceTest);
  //  interfacecombo->addStatusTest(interfaceTest);
  //  //interfacecombo->addStatusTest(interfaceTestUpdate);
  //
  //  converged->addStatusTest(interfacecombo);

  // setup tests for fluid velocities

  std::vector<Teuchos::RCP<const Epetra_Map>> fluidvel;
  fluidvel.push_back(FluidField()->InnerVelocityRowMap());
  fluidvel.push_back(Teuchos::null);
  CORE::LINALG::MultiMapExtractor fluidvelextract(*DofRowMap(), fluidvel);

  Teuchos::RCP<::NOX::StatusTest::Combo> fluidvelcombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> innerFluidVel = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "velocity", fluidvelextract, 0, nlParams.get<double>("Norm abs vel"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> innerFluidVelUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("velocity update", fluidvelextract, 0,
          nlParams.get<double>("Norm abs vel"), NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(innerFluidVel);
  fluidvelcombo->addStatusTest(innerFluidVel);
  // fluidvelcombo->addStatusTest(innerFluidVelUpdate);

  converged->addStatusTest(fluidvelcombo);

  // setup tests for fluid pressure

  std::vector<Teuchos::RCP<const Epetra_Map>> fluidpress;
  fluidpress.push_back(FluidField()->PressureRowMap());
  fluidpress.push_back(Teuchos::null);
  CORE::LINALG::MultiMapExtractor fluidpressextract(*DofRowMap(), fluidpress);

  Teuchos::RCP<::NOX::StatusTest::Combo> fluidpresscombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> fluidPress = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "pressure", fluidpressextract, 0, nlParams.get<double>("Norm abs pres"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> fluidPressUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("pressure update", fluidpressextract, 0,
          nlParams.get<double>("Norm abs pres"), NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(fluidPress);
  fluidpresscombo->addStatusTest(fluidPress);
  // fluidpresscombo->addStatusTest(fluidPressUpdate);

  // setup tests for volume constraint

  std::vector<Teuchos::RCP<const Epetra_Map>> volconstr;
  volconstr.push_back(conman_->GetConstraintMap());
  volconstr.push_back(Teuchos::null);
  CORE::LINALG::MultiMapExtractor volconstrextract(*DofRowMap(), volconstr);

  Teuchos::RCP<::NOX::StatusTest::Combo> volconstrcombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> VolConstr = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "constraints", volconstrextract, 0, nlParams.get<double>("Norm abs vol constr"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> VolConstrUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("constraints update", volconstrextract, 0,
          nlParams.get<double>("Norm abs vol constr"), NOX::FSI::PartialNormUpdate::Scaled));

  AddStatusTest(VolConstr);
  volconstrcombo->addStatusTest(VolConstr);
  // volconstrcombo->addStatusTest(VolConstrUpdate);


  converged->addStatusTest(volconstrcombo);


  return combo;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::ConstrMonolithic::CreateSystemMatrix(bool structuresplit)
{
  const Teuchos::ParameterList& fsidyn = GLOBAL::Problem::Instance()->FSIDynamicParams();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");

  // get the PCITER from inputfile
  std::vector<int> pciter;
  std::vector<double> pcomega;
  std::vector<int> spciter;
  std::vector<double> spcomega;
  std::vector<int> fpciter;
  std::vector<double> fpcomega;
  std::vector<int> apciter;
  std::vector<double> apcomega;
  {
    std::string word1;
    std::string word2;
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsimono, "PCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsimono, "PCOMEGA"));
      while (pciterstream >> word1) pciter.push_back(std::atoi(word1.c_str()));
      while (pcomegastream >> word2) pcomega.push_back(std::atof(word2.c_str()));
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsimono, "STRUCTPCITER"));
      std::istringstream pcomegastream(
          Teuchos::getNumericStringParameter(fsimono, "STRUCTPCOMEGA"));
      while (pciterstream >> word1) spciter.push_back(std::atoi(word1.c_str()));
      while (pcomegastream >> word2) spcomega.push_back(std::atof(word2.c_str()));
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsimono, "FLUIDPCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsimono, "FLUIDPCOMEGA"));
      while (pciterstream >> word1) fpciter.push_back(std::atoi(word1.c_str()));
      while (pcomegastream >> word2) fpcomega.push_back(std::atof(word2.c_str()));
    }
    {
      std::istringstream pciterstream(Teuchos::getNumericStringParameter(fsimono, "ALEPCITER"));
      std::istringstream pcomegastream(Teuchos::getNumericStringParameter(fsimono, "ALEPCOMEGA"));
      while (pciterstream >> word1) apciter.push_back(std::atoi(word1.c_str()));
      while (pcomegastream >> word2) apcomega.push_back(std::atof(word2.c_str()));
    }
  }

  //-----------------------------------------------------------------------------
  // create block system matrix
  //-----------------------------------------------------------------------------

  switch (linearsolverstrategy_)
  {
    case INPAR::FSI::PreconditionedKrylov:
      systemmatrix_ = Teuchos::rcp(new ConstrOverlappingBlockMatrix(Extractor(), *StructureField(),
          *FluidField(), *AleField(), structuresplit,
          CORE::UTILS::IntegralValue<int>(fsimono, "SYMMETRICPRECOND"), pcomega[0], pciter[0],
          spcomega[0], spciter[0], fpcomega[0], fpciter[0], apcomega[0], apciter[0]));

      break;
    default:
      FOUR_C_THROW("Unsupported type of monolithic solver/preconditioner!");
      break;
  }
}

FOUR_C_NAMESPACE_CLOSE
