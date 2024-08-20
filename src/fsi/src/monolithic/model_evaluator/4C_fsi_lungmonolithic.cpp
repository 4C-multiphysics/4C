/*----------------------------------------------------------------------*/
/*! \file
\brief Volume-coupled FSI (base class)

\level 3

*/
/*----------------------------------------------------------------------*/
#include "4C_fsi_lungmonolithic.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_fld_lung.hpp"
#include "4C_adapter_str_lung.hpp"
#include "4C_ale_utils_mapextractor.hpp"
#include "4C_constraint_dofset.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_fluid_utils_mapextractor.hpp"
#include "4C_fsi_lung_overlapprec.hpp"
#include "4C_fsi_monolithic_linearsystem.hpp"
#include "4C_fsi_statustest.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::LungMonolithic::LungMonolithic(
    const Epetra_Comm& comm, const Teuchos::ParameterList& timeparams)
    : BlockMonolithic(comm, timeparams)
{
  icoupfa_ = Teuchos::rcp(new Coupling::Adapter::Coupling());
  coupsaout_ = Teuchos::rcp(new Coupling::Adapter::Coupling());
  coupfsout_ = Teuchos::rcp(new Coupling::Adapter::Coupling());
  coupfaout_ = Teuchos::rcp(new Coupling::Adapter::Coupling());

  //-----------------------------------------------------------------------------
  // additional fluid-structure volume constraints
  //-----------------------------------------------------------------------------

  // Since the current design of the constraint manager and fsi
  // algorithms complicates the neat combination of both
  // (e.g. concerning the question of who owns what actually) in this
  // special application, the general functionality of the constraint
  // manager is included here on the algorithm level (as far as
  // needed).

  Teuchos::RCP<Adapter::FluidLung> fluidfield =
      Teuchos::rcp_dynamic_cast<Adapter::FluidLung>(fluid_field());
  const Teuchos::RCP<Adapter::StructureLung>& structfield =
      Teuchos::rcp_dynamic_cast<Adapter::StructureLung>(structure_field());

  // consistency check: all dofs contained in ale(fluid)-structure coupling need to
  // be part of the structure volume constraint, too. this needs to be checked because during
  // setup_system_matrix, we rely on this information!
  const Teuchos::RCP<const Epetra_Map> asimap = structure_field()->interface()->lung_asi_cond_map();
  for (int i = 0; i < asimap->NumMyElements(); ++i)
  {
    if (structfield->lung_constr_map()->LID(asimap->GID(i)) == -1)
      FOUR_C_THROW("dof of asi coupling is not contained in enclosing boundary");
  }

  std::set<int> FluidLungVolConIDs;
  std::set<int> StructLungVolConIDs;
  int FluidMinLungVolConID;
  int StructMinLungVolConID;

  structfield->list_lung_vol_cons(StructLungVolConIDs, StructMinLungVolConID);
  fluidfield->list_lung_vol_cons(FluidLungVolConIDs, FluidMinLungVolConID);

  // We want to be sure that both fluid and structure fields hold the
  // same constraint IDs. After all, every airway outlet needs to be
  // coupled to one structural volume. Therefore, merely comparing the
  // overall number and the minimum constraint ID is not sufficient here.

  for (std::set<int>::iterator iter = FluidLungVolConIDs.begin(); iter != FluidLungVolConIDs.end();
       ++iter)
  {
    if (StructLungVolConIDs.find(*iter) == StructLungVolConIDs.end())
      FOUR_C_THROW("No matching in fluid and structure lung volume constraints");
  }

  NumConstrID_ = FluidLungVolConIDs.size();

  ConstrDofSet_ = Teuchos::rcp(new CONSTRAINTS::ConstraintDofSet());
  ConstrDofSet_->assign_degrees_of_freedom(fluid_field()->discretization(), NumConstrID_, 0);

  // The "OffsetID" is used during the evaluation of constraints on
  // the element level. For assembly of the constraint parts, the gid
  // of the constraint dof (= Lagrange multiplier) needs to be known.
  //
  // gid = current constraint ID - minimum constraint ID + first gid of all constraints
  //                             \__________________________  ________________________/
  //                                                        \/
  //                                                   - OffsetID_
  //
  // By including the minimum constraint ID, one allows also to define
  // a set of constraints not starting from 1 in the input file.
  // Since the "OffsetID" is subtracted later on, we save its negative
  // value here.

  OffsetID_ = FluidMinLungVolConID - ConstrDofSet_->first_gid();
  ConstrMap_ = Teuchos::rcp(new Epetra_Map(*(ConstrDofSet_->dof_row_map())));

  // build an all reduced version of the constraintmap, since sometimes all processors
  // have to know all values of the constraints and Lagrange multipliers
  RedConstrMap_ = Core::LinAlg::allreduce_e_map(*ConstrMap_);

  // create importer
  ConstrImport_ = Teuchos::rcp(new Epetra_Export(*RedConstrMap_, *ConstrMap_));

  // initialize associated matrices and vectors

  // NOTE: everything that is determined in the fluid adapter needs to
  // be based on the fluid dofmap, i.e. also the ale related stuff!
  // Corresponding matrices then need to be transformed to the ale
  // dofmap using corresponding matrix transformators.

  LagrMultVec_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  LagrMultVecOld_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  IncLagrMultVec_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));

  // build merged structure dof map
  Teuchos::RCP<Epetra_Map> FullStructDofMap =
      Core::LinAlg::merge_map(*structure_field()->dof_row_map(), *ConstrMap_, false);
  Core::LinAlg::MapExtractor StructConstrExtractor(
      *FullStructDofMap, ConstrMap_, structure_field()->dof_row_map());

  AddStructConstrMatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          StructConstrExtractor, StructConstrExtractor, 81, false, true));

  AddFluidShapeDerivMatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *fluid_field()->interface(), *fluid_field()->interface(), 108, false, true));
  FluidConstrMatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(
      *fluid_field()->discretization()->dof_row_map(), NumConstrID_, false, true));
  ConstrFluidMatrix_ = Teuchos::rcp(new Core::LinAlg::SparseMatrix(*ConstrMap_,
      fluid_field()->discretization()->dof_row_map()->NumGlobalElements(), false, true));

  // additional "ale" matrices filled in the fluid elements
  Teuchos::RCP<Epetra_Map> emptymap =
      Teuchos::rcp(new Epetra_Map(-1, 0, nullptr, 0, fluid_field()->discretization()->get_comm()));
  Core::LinAlg::MapExtractor constrextractor;
  constrextractor.setup(*ConstrMap_, emptymap, ConstrMap_);
  AleConstrMatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          constrextractor, *fluid_field()->interface(), 108, false, true));
  ConstrAleMatrix_ =
      Teuchos::rcp(new Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>(
          *fluid_field()->interface(), constrextractor, 108, false, true));

  AddStructRHS_ =
      Teuchos::rcp(new Epetra_Vector(*structure_field()->discretization()->dof_row_map(), true));
  AddFluidRHS_ =
      Teuchos::rcp(new Epetra_Vector(*fluid_field()->discretization()->dof_row_map(), true));
  ConstrRHS_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));

  OldVols_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  CurrVols_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  SignVolsRed_ = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_, true));
  dVstruct_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));

  OldFlowRates_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  CurrFlowRates_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));
  dVfluid_ = Teuchos::rcp(new Epetra_Vector(*ConstrMap_, true));

  // time integration factor for flow rates
  theta_ = 0.5;

  // determine initial volumes of parenchyma balloons
  Teuchos::RCP<Epetra_Vector> OldVolsRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));

  structfield->initialize_vol_con(OldVolsRed, SignVolsRed_, OffsetID_);

  OldVols_->PutScalar(0.0);
  OldVols_->Export(*OldVolsRed, *ConstrImport_, Add);

  // determine initial flow rates at outlets
  Teuchos::RCP<Epetra_Vector> OldFlowRatesRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));

  fluidfield->initialize_vol_con(OldFlowRatesRed, OffsetID_);

  OldFlowRates_->PutScalar(0.0);
  OldFlowRates_->Export(*OldFlowRatesRed, *ConstrImport_, Add);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::general_setup()
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  linearsolverstrategy_ =
      Core::UTILS::integral_value<Inpar::FSI::LinearBlockSolver>(fsimono, "LINEARBLOCKSOLVER");

  set_default_parameters(fsidyn, nox_parameter_list());

  // ToDo: Set more detailed convergence tolerances like in standard FSI
  // additionally set tolerance for volume constraint
  nox_parameter_list().set("Norm abs vol constr", fsimono.get<double>("CONVTOL"));

  //-----------------------------------------------------------------------------
  // ordinary fsi coupling
  //-----------------------------------------------------------------------------

  // right now we use matching meshes at the interface

  Coupling::Adapter::Coupling& coupsf = structure_fluid_coupling();
  Coupling::Adapter::Coupling& coupsa = structure_ale_coupling();
  Coupling::Adapter::Coupling& coupfa = fluid_ale_coupling();

  const int ndim = Global::Problem::instance()->n_dim();

  // structure to fluid

  coupsf.setup_condition_coupling(*structure_field()->discretization(),
      structure_field()->interface()->fsi_cond_map(), *fluid_field()->discretization(),
      fluid_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  // structure to ale

  coupsa.setup_condition_coupling(*structure_field()->discretization(),
      structure_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  // fluid to ale at the interface

  icoupfa_->setup_condition_coupling(*fluid_field()->discretization(),
      fluid_field()->interface()->fsi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->fsi_cond_map(), "FSICoupling", ndim);

  // In the following we assume that both couplings find the same dof
  // map at the structural side. This enables us to use just one
  // interface dof map for all fields and have just one transfer
  // operator from the interface map to the full field map.
  if (not coupsf.master_dof_map()->SameAs(*coupsa.master_dof_map()))
    FOUR_C_THROW("structure interface dof maps do not match");

  if (coupsf.master_dof_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("No nodes in matching FSI interface. Empty FSI coupling condition?");

  // the fluid-ale coupling always matches
  const Epetra_Map* fluidnodemap = fluid_field()->discretization()->node_row_map();
  const Epetra_Map* alenodemap = ale_field()->discretization()->node_row_map();

  coupfa.setup_coupling(*fluid_field()->discretization(), *ale_field()->discretization(),
      *fluidnodemap, *alenodemap, ndim);

  fluid_field()->set_mesh_map(coupfa.master_dof_map());

  aleresidual_ = Teuchos::rcp(new Epetra_Vector(*ale_field()->interface()->Map(0)));

  //-----------------------------------------------------------------------------
  // additional coupling of structure and ale field at the outflow boundary
  //-----------------------------------------------------------------------------

  // coupling of structure and ale dofs at airway outflow
  coupsaout_->setup_constrained_condition_coupling(*structure_field()->discretization(),
      structure_field()->interface()->lung_asi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->lung_asi_cond_map(), "StructAleCoupling", "FSICoupling", ndim);
  if (coupsaout_->master_dof_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("No nodes in matching structure ale interface. Empty coupling condition?");

  // coupling of fluid and structure dofs at airway outflow
  coupfsout_->setup_constrained_condition_coupling(*fluid_field()->discretization(),
      fluid_field()->interface()->lung_asi_cond_map(), *structure_field()->discretization(),
      structure_field()->interface()->lung_asi_cond_map(), "StructAleCoupling", "FSICoupling",
      ndim);
  if (coupfsout_->master_dof_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("No nodes in matching structure ale/fluid interface. Empty coupling condition?");

  // coupling of fluid and ale dofs at airway outflow
  coupfaout_->setup_constrained_condition_coupling(*fluid_field()->discretization(),
      fluid_field()->interface()->lung_asi_cond_map(), *ale_field()->discretization(),
      ale_field()->interface()->lung_asi_cond_map(), "StructAleCoupling", "FSICoupling", ndim);
  if (coupfaout_->master_dof_map()->NumGlobalElements() == 0)
    FOUR_C_THROW("No nodes in matching ale fluid ouflow interface. Empty coupling condition?");

  //-----------------------------------------------------------------------------
  // enable output of changes in volumes in text file
  //-----------------------------------------------------------------------------

  if (get_comm().MyPID() == 0)
  {
    std::string outputprefix =
        Global::Problem::instance()->output_control_file()->new_output_file_name();
    std::string dfluidfilename;
    std::string dstructfilename;
    std::string absstructfilename;
    std::string absfluidfilename;
    size_t posn = outputprefix.rfind('-');
    if (posn != std::string::npos)
    {
      std::string number = outputprefix.substr(posn + 1);
      std::string prefix = outputprefix.substr(0, posn);
      std::ostringstream sf;
      sf << prefix << "_dVfluid"
         << "-" << number << ".txt";
      dfluidfilename = sf.str();
      std::ostringstream ss;
      ss << prefix << "_dVstruct"
         << "-" << number << ".txt";
      dstructfilename = ss.str();
      std::ostringstream sas;
      sas << prefix << "_absVstruct"
          << "-" << number << ".txt";
      absstructfilename = sas.str();
    }
    else
    {
      std::ostringstream sf;
      sf << outputprefix << "_dVfluid.txt";
      dfluidfilename = sf.str();
      std::ostringstream ss;
      ss << outputprefix << "_dVstruct.txt";
      dstructfilename = ss.str();
      std::ostringstream sas;
      sas << outputprefix << "_absVstruct.txt";
      absstructfilename = sas.str();
    }

    outfluiddvol_.open(dfluidfilename.c_str());
    outstructdvol_.open(dstructfilename.c_str());
    outstructabsvol_.open(absstructfilename.c_str());
  }

  writerestartevery_ = fsidyn.get<int>("RESTARTEVRY");

  // ToDo: Setup the monolithic DBC map extractor and use only this to handle DBCs in Matrix and RHS
  dbcmaps_ = Teuchos::null;

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::LungMonolithic::struct_to_ale_outflow(
    Teuchos::RCP<Epetra_Vector> iv) const
{
  return coupsaout_->master_to_slave(iv);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::evaluate(Teuchos::RCP<const Epetra_Vector> step_increment)
{
  //-----------------------------------------------------------------------------
  // evaluation of all fields
  //-----------------------------------------------------------------------------

  FSI::Monolithic::evaluate(step_increment);

  //-----------------------------------------------------------------------------
  // evaluation of lung volume constraints
  //-----------------------------------------------------------------------------

  if (step_increment != Teuchos::null)
  {
    // extract sum of all iterative increments of lagrange multipliers
    // in this time step (this is what we get from NOX)
    IncLagrMultVec_ = extractor().extract_vector(step_increment, 3);

    // update current lagrange multipliers
    LagrMultVec_->Update(1.0, *LagrMultVecOld_, 1.0, *IncLagrMultVec_, 0.0);
  }

  //-----------------------------------------------------------------------------
  // structure part

  // create redundant vectors
  Teuchos::RCP<Epetra_Vector> LagrMultVecRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Core::LinAlg::export_to(*LagrMultVec_, *LagrMultVecRed);
  Teuchos::RCP<Epetra_Vector> CurrVolsRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));

  const Teuchos::RCP<Adapter::StructureLung>& structfield =
      Teuchos::rcp_dynamic_cast<Adapter::StructureLung>(structure_field());
  CurrVolsRed->PutScalar(0.0);
  AddStructRHS_->PutScalar(0.0);
  AddStructConstrMatrix_->zero();

  structfield->evaluate_vol_con(
      AddStructConstrMatrix_, AddStructRHS_, CurrVolsRed, SignVolsRed_, LagrMultVecRed, OffsetID_);

  // Export redundant vector into distributed one
  CurrVols_->PutScalar(0.0);
  CurrVols_->Export(*CurrVolsRed, *ConstrImport_, Add);


  // negative sign (for shift to rhs) is already taken into account!
  dVstruct_->Update(1.0, *CurrVols_, -1.0, *OldVols_, 0.0);
  ConstrRHS_->Update(-1.0, *dVstruct_, 0.0);

  //-----------------------------------------------------------------------------
  // fluid/ale part

  Teuchos::RCP<Adapter::FluidLung> fluidfield =
      Teuchos::rcp_dynamic_cast<Adapter::FluidLung>(fluid_field());

  // create redundant vector
  Teuchos::RCP<Epetra_Vector> CurrFlowRatesRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));

  CurrFlowRatesRed->PutScalar(0.0);
  AddFluidRHS_->PutScalar(0.0);
  AddFluidShapeDerivMatrix_->zero();
  FluidConstrMatrix_->zero();
  ConstrFluidMatrix_->zero();
  AleConstrMatrix_->zero();
  ConstrAleMatrix_->zero();

  const double dt = LungMonolithic::dt();
  const double dttheta = dt * theta_;

  fluidfield->evaluate_vol_con(AddFluidShapeDerivMatrix_, FluidConstrMatrix_, ConstrFluidMatrix_,
      AleConstrMatrix_, ConstrAleMatrix_, AddFluidRHS_, CurrFlowRatesRed, LagrMultVecRed, OffsetID_,
      dttheta);

  // Export redundant vector into distributed one
  CurrFlowRates_->PutScalar(0.0);
  CurrFlowRates_->Export(*CurrFlowRatesRed, *ConstrImport_, Add);

  // negative sign (for shift to rhs) is already taken into account!
  dVfluid_->Update(dt * theta_, *CurrFlowRates_, dt * (1.0 - theta_), *OldFlowRates_, 0.0);
  ConstrRHS_->Update(1.0, *dVfluid_, 1.0);

  //   std::cout << "CurrFlowRates_:\n" << *CurrFlowRates_ << std::endl;
  //   std::cout << "CurrVols_:\n" << *CurrVols_ << std::endl;
  //   std::cout << "LagrMultVec_:\n" << *LagrMultVec_ << std::endl;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::scale_system(Core::LinAlg::BlockSparseMatrixBase& mat, Epetra_Vector& b)
{
  // should we scale the system?
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = (bool)Core::UTILS::integral_value<int>(fsimono, "INFNORMSCALING");

  if (scaling_infnorm)
  {
    // The matrices are modified here. Do we have to change them back later on?

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.matrix(0, 0).epetra_matrix();
    srowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    scolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*srowsum_);
    A->InvColSums(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.matrix(0, 1).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 2).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 3).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(1, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(2, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(3, 0).epetra_matrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.matrix(2, 2).epetra_matrix();
    arowsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    acolsum_ = Teuchos::rcp(new Epetra_Vector(A->RowMap(), false));
    A->InvRowSums(*arowsum_);
    A->InvColSums(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.matrix(2, 0).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 1).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 3).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(0, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(1, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(3, 2).epetra_matrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");

    Teuchos::RCP<Epetra_Vector> sx = extractor().extract_vector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = extractor().extract_vector(b, 2);

    if (sx->Multiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->Multiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::unscale_solution(
    Core::LinAlg::BlockSparseMatrixBase& mat, Epetra_Vector& x, Epetra_Vector& b)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
  const Teuchos::ParameterList& fsimono = fsidyn.sublist("MONOLITHIC SOLVER");
  const bool scaling_infnorm = (bool)Core::UTILS::integral_value<int>(fsimono, "INFNORMSCALING");

  if (scaling_infnorm)
  {
    Teuchos::RCP<Epetra_Vector> sy = extractor().extract_vector(x, 0);
    Teuchos::RCP<Epetra_Vector> ay = extractor().extract_vector(x, 2);

    if (sy->Multiply(1.0, *scolsum_, *sy, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ay->Multiply(1.0, *acolsum_, *ay, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sy, 0, x);
    extractor().insert_vector(*ay, 2, x);

    Teuchos::RCP<Epetra_Vector> sx = extractor().extract_vector(b, 0);
    Teuchos::RCP<Epetra_Vector> ax = extractor().extract_vector(b, 2);

    if (sx->ReciprocalMultiply(1.0, *srowsum_, *sx, 0.0)) FOUR_C_THROW("structure scaling failed");
    if (ax->ReciprocalMultiply(1.0, *arowsum_, *ax, 0.0)) FOUR_C_THROW("ale scaling failed");

    extractor().insert_vector(*sx, 0, b);
    extractor().insert_vector(*ax, 2, b);

    Teuchos::RCP<Epetra_CrsMatrix> A = mat.matrix(0, 0).epetra_matrix();
    srowsum_->Reciprocal(*srowsum_);
    scolsum_->Reciprocal(*scolsum_);
    if (A->LeftScale(*srowsum_) or A->RightScale(*scolsum_) or
        mat.matrix(0, 1).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 2).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(0, 3).epetra_matrix()->LeftScale(*srowsum_) or
        mat.matrix(1, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(2, 0).epetra_matrix()->RightScale(*scolsum_) or
        mat.matrix(3, 0).epetra_matrix()->RightScale(*scolsum_))
      FOUR_C_THROW("structure scaling failed");

    A = mat.matrix(2, 2).epetra_matrix();
    arowsum_->Reciprocal(*arowsum_);
    acolsum_->Reciprocal(*acolsum_);
    if (A->LeftScale(*arowsum_) or A->RightScale(*acolsum_) or
        mat.matrix(2, 0).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 1).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(2, 3).epetra_matrix()->LeftScale(*arowsum_) or
        mat.matrix(0, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(1, 2).epetra_matrix()->RightScale(*acolsum_) or
        mat.matrix(3, 2).epetra_matrix()->RightScale(*acolsum_))
      FOUR_C_THROW("ale scaling failed");
  }

  // very simple hack just to see the linear solution

  Epetra_Vector r(b.Map());
  mat.Apply(x, r);
  r.Update(1., b, 1.);

  Teuchos::RCP<Epetra_Vector> sr = extractor().extract_vector(r, 0);
  Teuchos::RCP<Epetra_Vector> fr = extractor().extract_vector(r, 1);
  Teuchos::RCP<Epetra_Vector> ar = extractor().extract_vector(r, 2);
  Teuchos::RCP<Epetra_Vector> cr = extractor().extract_vector(r, 3);

  // increment additional ale residual
  aleresidual_->Update(-1., *ar, 0.);

  std::ios_base::fmtflags flags = utils()->out().flags();

  double n, ns, nf, na, nc;
  r.Norm2(&n);
  sr->Norm2(&ns);
  fr->Norm2(&nf);
  ar->Norm2(&na);
  cr->Norm2(&nc);
  utils()->out() << std::scientific << "\nlinear solver quality:\n"
                 << "L_2-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "   |rc|=" << nc << "\n";
  r.NormInf(&n);
  sr->NormInf(&ns);
  fr->NormInf(&nf);
  ar->NormInf(&na);
  cr->NormInf(&nc);
  utils()->out() << "L_inf-norms:\n"
                 << "   |r|=" << n << "   |rs|=" << ns << "   |rf|=" << nf << "   |ra|=" << na
                 << "   |rc|=" << nc << "\n";

  utils()->out().flags(flags);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::Epetra::LinearSystem> FSI::LungMonolithic::create_linear_system(
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
    case Inpar::FSI::PreconditionedKrylov:
      linSys = Teuchos::rcp(new  // ::NOX::Epetra::LinearSystemAztecOO(
          FSI::MonolithicLinearSystem(printParams, *lsParams, Teuchos::rcp(iJac, false), J,
              Teuchos::rcp(iPrec, false), M, noxSoln));
      break;
    default:
      FOUR_C_THROW("Unsupported type of monolithic solver/preconditioner!");
      break;
  }

  return linSys;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<::NOX::StatusTest::Combo> FSI::LungMonolithic::create_status_test(
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
      "displacement", extractor(), 0, nlParams.get<double>("Norm abs disp"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> structureDispUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("displacement update", extractor(), 0,
          nlParams.get<double>("Norm abs disp"), NOX::FSI::PartialNormUpdate::Scaled));

  add_status_test(structureDisp);
  structcombo->addStatusTest(structureDisp);

  converged->addStatusTest(structcombo);

  // setup tests for interface

  std::vector<Teuchos::RCP<const Epetra_Map>> interface;
  interface.push_back(fluid_field()->interface()->fsi_cond_map());
  interface.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor interfaceextract(*dof_row_map(), interface);

  Teuchos::RCP<::NOX::StatusTest::Combo> interfacecombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> interfaceTest = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "interface", interfaceextract, 0, nlParams.get<double>("Norm abs vel"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> interfaceTestUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("interface update", interfaceextract, 0,
          nlParams.get<double>("Norm abs vel"), NOX::FSI::PartialNormUpdate::Scaled));

  add_status_test(interfaceTest);
  interfacecombo->addStatusTest(interfaceTest);

  converged->addStatusTest(interfacecombo);

  // setup tests for fluid velocities

  std::vector<Teuchos::RCP<const Epetra_Map>> fluidvel;
  fluidvel.push_back(fluid_field()->inner_velocity_row_map());
  fluidvel.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor fluidvelextract(*dof_row_map(), fluidvel);

  Teuchos::RCP<::NOX::StatusTest::Combo> fluidvelcombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> innerFluidVel = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "velocity", fluidvelextract, 0, nlParams.get<double>("Norm abs vel"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> innerFluidVelUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("velocity update", fluidvelextract, 0,
          nlParams.get<double>("Norm abs vel"), NOX::FSI::PartialNormUpdate::Scaled));

  add_status_test(innerFluidVel);
  fluidvelcombo->addStatusTest(innerFluidVel);

  converged->addStatusTest(fluidvelcombo);

  // setup tests for fluid pressure

  std::vector<Teuchos::RCP<const Epetra_Map>> fluidpress;
  fluidpress.push_back(fluid_field()->pressure_row_map());
  fluidpress.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor fluidpressextract(*dof_row_map(), fluidpress);

  Teuchos::RCP<::NOX::StatusTest::Combo> fluidpresscombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> fluidPress = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "pressure", fluidpressextract, 0, nlParams.get<double>("Norm abs pres"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> fluidPressUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("pressure update", fluidpressextract, 0,
          nlParams.get<double>("Norm abs pres"), NOX::FSI::PartialNormUpdate::Scaled));

  add_status_test(fluidPress);
  fluidpresscombo->addStatusTest(fluidPress);

  converged->addStatusTest(fluidpresscombo);

  // setup tests for volume constraint

  std::vector<Teuchos::RCP<const Epetra_Map>> volconstr;
  volconstr.push_back(ConstrMap_);
  volconstr.push_back(Teuchos::null);
  Core::LinAlg::MultiMapExtractor volconstrextract(*dof_row_map(), volconstr);

  Teuchos::RCP<::NOX::StatusTest::Combo> volconstrcombo =
      Teuchos::rcp(new ::NOX::StatusTest::Combo(::NOX::StatusTest::Combo::OR));

  Teuchos::RCP<NOX::FSI::PartialNormF> VolConstr = Teuchos::rcp(new NOX::FSI::PartialNormF(
      "volume constraint", volconstrextract, 0, nlParams.get<double>("Norm abs vol constr"),
      ::NOX::Abstract::Vector::TwoNorm, NOX::FSI::PartialNormF::Scaled));
  Teuchos::RCP<NOX::FSI::PartialNormUpdate> VolConstrUpdate =
      Teuchos::rcp(new NOX::FSI::PartialNormUpdate("volume constraint update", volconstrextract, 0,
          nlParams.get<double>("Norm abs vol constr"), NOX::FSI::PartialNormUpdate::Scaled));

  add_status_test(VolConstr);
  volconstrcombo->addStatusTest(VolConstr);

  converged->addStatusTest(volconstrcombo);

  return combo;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::update()
{
  FSI::BlockMonolithic::update();

  // update fluid flow rates and structure volumes and lagrange multipliers
  OldVols_->Update(1.0, *CurrVols_, 0.0);
  OldFlowRates_->Update(1.0, *CurrFlowRates_, 0.0);
  LagrMultVecOld_->Update(1.0, *LagrMultVec_, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::output()
{
  // Note: The order is important here! In here control file entries are
  // written. And these entries define the order in which the filters handle
  // the Discretizations, which in turn defines the dof number ordering of the
  // Discretizations.
  structure_field()->output();

  // additional output of volume constraint related forces
  //   Adapter::StructureLung& structfield =
  //   dynamic_cast<Adapter::StructureLung&>(structure_field());
  //   structfield.OutputForces(AddStructRHS_);

  // Write history vectors in case of restart
  // This is done using the structure DiscretizationWriter, hence it
  // is placed in between the output of the single fields.
  if (writerestartevery_ and (step() % writerestartevery_ == 0))
  {
    const Teuchos::RCP<Adapter::StructureLung>& structfield =
        Teuchos::rcp_dynamic_cast<Adapter::StructureLung>(structure_field());
    Teuchos::RCP<Epetra_Vector> OldFlowRatesRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
    Core::LinAlg::export_to(*OldFlowRates_, *OldFlowRatesRed);
    Teuchos::RCP<Epetra_Vector> OldVolsRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
    Core::LinAlg::export_to(*OldVols_, *OldVolsRed);
    Teuchos::RCP<Epetra_Vector> LagrMultVecOldRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
    Core::LinAlg::export_to(*LagrMultVecOld_, *LagrMultVecOldRed);
    structfield->write_vol_con_restart(OldFlowRatesRed, OldVolsRed, LagrMultVecOldRed);
  }

  fluid_field()->output();

  // additional output of volume constraint related forces
  //   Adapter::FluidLung& fluidfield = dynamic_cast<Adapter::FluidLung&>(fluid_field());
  //   fluidfield->OutputForces(AddFluidRHS_);

  ale_field()->output();

  // output of volumes for visualization (e.g. gnuplot)

  Teuchos::RCP<Epetra_Vector> dVfluidRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Core::LinAlg::export_to(*dVfluid_, *dVfluidRed);

  if (get_comm().MyPID() == 0)
  {
    outfluiddvol_ << step();
    for (int i = 0; i < dVfluidRed->MyLength(); ++i)
    {
      outfluiddvol_ << "\t" << (*dVfluidRed)[i];
    }
    outfluiddvol_ << "\n" << std::flush;
  }

  Teuchos::RCP<Epetra_Vector> dVstructRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Core::LinAlg::export_to(*dVstruct_, *dVstructRed);

  if (get_comm().MyPID() == 0)
  {
    outstructdvol_ << step();
    for (int i = 0; i < dVstructRed->MyLength(); ++i)
    {
      outstructdvol_ << "\t" << (*dVstructRed)[i];
    }
    outstructdvol_ << "\n" << std::flush;
  }

  Teuchos::RCP<Epetra_Vector> VstructRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Core::LinAlg::export_to(*CurrVols_, *VstructRed);

  if (get_comm().MyPID() == 0)
  {
    outstructabsvol_ << step();
    for (int i = 0; i < VstructRed->MyLength(); ++i)
    {
      outstructabsvol_ << "\t" << (*VstructRed)[i];
    }
    outstructabsvol_ << "\n" << std::flush;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::read_restart(int step)
{
  FSI::Monolithic::read_restart(step);

  const Teuchos::RCP<Adapter::StructureLung>& structfield =
      Teuchos::rcp_dynamic_cast<Adapter::StructureLung>(structure_field());

  Teuchos::RCP<Epetra_Vector> OldFlowRatesRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Teuchos::RCP<Epetra_Vector> OldVolsRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));
  Teuchos::RCP<Epetra_Vector> OldLagrMultRed = Teuchos::rcp(new Epetra_Vector(*RedConstrMap_));

  structfield->read_vol_con_restart(step, OldFlowRatesRed, OldVolsRed, OldLagrMultRed);

  // Export redundant vector into distributed one
  OldVols_->PutScalar(0.0);
  OldVols_->Export(*OldVolsRed, *ConstrImport_, Insert);
  CurrVols_->Update(1.0, *OldVols_, 0.0);
  OldFlowRates_->PutScalar(0.0);
  OldFlowRates_->Export(*OldFlowRatesRed, *ConstrImport_, Insert);
  CurrFlowRates_->Update(1.0, *OldFlowRates_, 0.0);
  LagrMultVecOld_->PutScalar(0.0);
  LagrMultVecOld_->Export(*OldLagrMultRed, *ConstrImport_, Insert);
  LagrMultVec_->Update(1.0, *LagrMultVecOld_, 0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::prepare_time_step()
{
  FSI::BlockMonolithic::prepare_time_step();

  // additional lung volume constraint stuff

  // Update of Lagrange multipliers, current volumes and flow rates is
  // not necessary here, since these values are already equal to the
  // "old" ones (cf. update()). Note that we assume a constant
  // predictor here!

  IncLagrMultVec_->PutScalar(0.0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> FSI::LungMonolithic::system_matrix() const
{
  return systemmatrix_;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::LungMonolithic::create_system_matrix(bool structuresplit)
{
  const Teuchos::ParameterList& fsidyn = Global::Problem::instance()->fsi_dynamic_params();
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
    case Inpar::FSI::PreconditionedKrylov:
      systemmatrix_ = Teuchos::rcp(new LungOverlappingBlockMatrix(extractor(), *structure_field(),
          *fluid_field(), *ale_field(), structuresplit,
          Core::UTILS::integral_value<int>(fsimono, "SYMMETRICPRECOND"), pcomega[0], pciter[0],
          spcomega[0], spciter[0], fpcomega[0], fpciter[0], apcomega[0], apciter[0]));
      break;
    default:
      FOUR_C_THROW("Unsupported type of monolithic solver");
      break;
  }
}

FOUR_C_NAMESPACE_CLOSE
