/*----------------------------------------------------------------------*/
/*! \file


\brief cpp-file associated with algorithmic routines for two-way coupled partitioned
       solution approaches to fluid-structure-scalar-scalar interaction
       (FS3I). Specifically related version for multiscale approches. This file thereby holds
       all functions related with the large time scale simulation and
       the small to large to small time scale 'communication'.

\level 3


----------------------------------------------------------------------*/


#include "baci_adapter_ale_fsi.hpp"
#include "baci_adapter_fld_fluid_ac_fsi.hpp"
#include "baci_adapter_str_fsiwrapper.hpp"
#include "baci_fluid_utils_mapextractor.hpp"
#include "baci_fs3i_ac_fsi.hpp"
#include "baci_fsi_monolithic.hpp"
#include "baci_global_data.hpp"
#include "baci_inpar_material.hpp"
#include "baci_io.hpp"
#include "baci_io_control.hpp"
#include "baci_lib_discret.hpp"
#include "baci_lib_element.hpp"
#include "baci_linalg_mapextractor.hpp"
#include "baci_linalg_utils_sparse_algebra_assemble.hpp"
#include "baci_linalg_utils_sparse_algebra_create.hpp"
#include "baci_linear_solver_method_linalg.hpp"
#include "baci_mat_growth.hpp"
#include "baci_mat_growth_law.hpp"
#include "baci_mat_material.hpp"
#include "baci_mat_par_bundle.hpp"
#include "baci_mat_so3_material.hpp"
#include "baci_scatra_algorithm.hpp"
#include "baci_scatra_timint_implicit.hpp"
#include "baci_structure_aux.hpp"

BACI_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | timeloop for small time scales                            Thon 07/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScaleLoop()
{
  PrepareLargeTimeScaleLoop();

  while (LargeTimeScaleLoopNotFinished())
  {
    LargeTimeScalePrepareTimeStep();

    LargeTimeScaleOuterLoop();

    LargeTimeScaleUpdateAndOutput();
  }

  FinishLargeTimeScaleLoop();
}

/*----------------------------------------------------------------------*
 | Prepare the large time scale loop                         Thon 08/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::PrepareLargeTimeScaleLoop()
{
  // print info
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n************************************************************************"
                 "\n                         LARGE TIME SCALE LOOP"
                 "\n************************************************************************"
              << std::endl;
  }
  // Set large time scale time step in both scatra fields
  scatravec_[0]->ScaTraField()->SetDt(dt_large_);
  scatravec_[1]->ScaTraField()->SetDt(dt_large_);

  // set mean values in scatra fields
  LargeTimeScaleSetFSISolution();

  // set back large time scale flags
  fsineedsupdate_ = false;
  growth_updates_counter_ = 0;

  // Save the phinp vector at the beginning of the large time scale loop in
  // in order to estimate the so far induced growth
  *structurephinp_blts_ = *scatravec_[1]->ScaTraField()->Phinp();
}

/*----------------------------------------------------------------------*
 |  Set mean wall shear stresses in scatra fields            Thon 11/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::SetMeanWallShearStresses() const
{
  std::vector<Teuchos::RCP<const Epetra_Vector>> wss;

  // ############ Fluid Field ###############
  scatravec_[0]->ScaTraField()->SetWallShearStresses(FluidToFluidScalar(WallShearStress_lp_));

  // ############ Structure Field ###############

  // extract FSI-Interface from fluid field
  Teuchos::RCP<Epetra_Vector> WallShearStress =
      fsi_->FluidField()->Interface()->ExtractFSICondVector(WallShearStress_lp_);

  // replace global fluid interface dofs through structure interface dofs
  WallShearStress = fsi_->FluidToStruct(WallShearStress);

  // insert structure interface entries into vector with full structure length
  Teuchos::RCP<Epetra_Vector> structurewss =
      CORE::LINALG::CreateVector(*(fsi_->StructureField()->Interface()->FullMap()), true);

  // Parameter int block of function InsertVector: (0: inner dofs of structure, 1: interface dofs of
  // structure, 2: inner dofs of porofluid, 3: interface dofs of porofluid )
  fsi_->StructureField()->Interface()->InsertVector(WallShearStress, 1, structurewss);
  scatravec_[1]->ScaTraField()->SetWallShearStresses(StructureToStructureScalar(structurewss));
}

/*----------------------------------------------------------------------*
 |  Set mean concentration of the fluid scatra field         Thon 11/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::SetMeanFluidScatraConcentration()
{
  Teuchos::RCP<const Epetra_Vector> MeanFluidConc = meanmanager_->GetMeanValue("mean_phi");

  scatravec_[0]->ScaTraField()->SetMeanConcentration(MeanFluidConc);
}

/*----------------------------------------------------------------------*
 |  Set zero velocity field in scatra fields                 Thon 11/14 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::SetZeroVelocityField()
{
  Teuchos::RCP<Epetra_Vector> zeros =
      Teuchos::rcp(new Epetra_Vector(fsi_->FluidField()->Velnp()->Map(), true));
  scatravec_[0]->ScaTraField()->SetVelocityField(
      FluidToFluidScalar(zeros), Teuchos::null, FluidToFluidScalar(zeros), Teuchos::null);
  Teuchos::RCP<Epetra_Vector> zeros2 =
      Teuchos::rcp(new Epetra_Vector(fsi_->StructureField()->Velnp()->Map(), true));
  scatravec_[1]->ScaTraField()->SetVelocityField(StructureToStructureScalar(zeros2), Teuchos::null,
      StructureToStructureScalar(zeros2), Teuchos::null);
}

/*-------------------------------------------------------------------------------*
 | Evaluate surface permeability condition for struct scatra field    Thon 08/15 |
 *-------------------------------------------------------------------------------*/
void FS3I::ACFSI::EvaluateithScatraSurfacePermeability(const int i  // id of scalar to evaluate
)
{
  // Note: 0 corresponds to fluid-scatra
  //      1 corresponds to structure-scatra

  //----------------------------------------------------------------------
  // set membrane concentrations
  //----------------------------------------------------------------------
  SetMembraneConcentration();

  //----------------------------------------------------------------------
  // evaluate simplified kedem-katchalsy condtion
  //----------------------------------------------------------------------
  Teuchos::RCP<Epetra_Vector> rhs_scal = scatracoupforce_[i];
  Teuchos::RCP<CORE::LINALG::SparseMatrix> mat_scal = scatracoupmat_[i];

  rhs_scal->PutScalar(0.0);
  mat_scal->Zero();

  scatravec_[i]->ScaTraField()->SurfacePermeability(mat_scal, rhs_scal);

  // apply Dirichlet boundary conditions to coupling matrix and vector
  const Teuchos::RCP<const Epetra_Map> dbcmap =
      scatravec_[i]->ScaTraField()->DirichMaps()->CondMap();
  mat_scal->ApplyDirichlet(*dbcmap, false);
  CORE::LINALG::ApplyDirichletToSystem(*rhs_scal, *scatrazeros_[i], *dbcmap);
}

/*----------------------------------------------------------------------*
 | Finish the large time scale loop                          Thon 08/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::FinishLargeTimeScaleLoop()
{
  // Set small time scale time step size
  scatravec_[0]->ScaTraField()->SetDt(dt_);
  scatravec_[1]->ScaTraField()->SetDt(dt_);

  // Fix time and step in fsi and fluid scatra field

  // We start the small time scale with a new cycle. But since dt_large is a
  // multiple of fsiperiod_ we are already at the this time_
  // We do not modify the step_ counter; we just keep counting..

  double tmp = fmod(time_, dt_large_);
  tmp = tmp - fmod(tmp, 10.0 * fsiperiod_) + 10.0 * fsiperiod_;
  time_ = tmp;

  SetTimeAndStepInFSI(time_, step_);
  scatravec_[0]->ScaTraField()->SetTimeStep(time_, step_);
  scatravec_[1]->ScaTraField()->SetTimeStep(time_, step_);

  // we now have to fix the time_ and step_ of the structure field, since this is not shifted
  // in PrepareTimeStep(), but in Update(), which we here will not call. So..
  fsi_->StructureField()->SetTime(time_);
  fsi_->StructureField()->SetTimen(time_ + fsi_->FluidField()->Dt());
  fsi_->StructureField()->SetStep(step_);
  fsi_->StructureField()->SetStepn(step_ + 1);

  // we start with a clean small time scale loop
  fsiisperiodic_ = false;
  scatraisperiodic_ = false;


  // NOTE: we start a new output file since paraview does only read floating point numbers.
  // Hence the upcoming small time scale calculation may not be displayable in paraview due
  // to the large variety in the time scales. Bad thing :(
  //*-------------------------------------------------------------------------------*
  // | create new output file
  //*-------------------------------------------------------------------------------*/
  Teuchos::RCP<IO::DiscretizationWriter> output_writer =
      GLOBAL::Problem::Instance()->GetDis("structure")->Writer();
  output_writer->NewResultFile(step_);
  // and write all meshes
  output_writer->CreateNewResultAndMeshFile();
  output_writer->WriteMesh(0, 0.0);
  output_writer = GLOBAL::Problem::Instance()->GetDis("fluid")->Writer();
  output_writer->CreateNewResultAndMeshFile();
  output_writer->WriteMesh(0, 0.0);
  output_writer = GLOBAL::Problem::Instance()->GetDis("ale")->Writer();
  output_writer->CreateNewResultAndMeshFile();
  output_writer->WriteMesh(0, 0.0);
  output_writer = GLOBAL::Problem::Instance()->GetDis("scatra1")->Writer();
  output_writer->CreateNewResultAndMeshFile();
  output_writer->WriteMesh(0, 0.0);
  output_writer = GLOBAL::Problem::Instance()->GetDis("scatra2")->Writer();
  output_writer->CreateNewResultAndMeshFile();
  output_writer->WriteMesh(0, 0.0);

  // write outputs in new file
  constexpr bool force_prepare = false;
  fsi_->PrepareOutput(force_prepare);

  FsiOutput();
  ScatraOutput();
}

/*----------------------------------------------------------------------*
 | timeloop for large time scales                            Thon 07/15 |
 *----------------------------------------------------------------------*/
bool FS3I::ACFSI::LargeTimeScaleLoopNotFinished() { return NotFinished() and not fsineedsupdate_; }

/*----------------------------------------------------------------------*
 | Prepare small time scale time step                        Thon 07/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScalePrepareTimeStep()
{
  // Set large time scale time step in both scatra fields
  scatravec_[0]->ScaTraField()->SetDt(dt_large_);
  scatravec_[1]->ScaTraField()->SetDt(dt_large_);

  // Increment time and step
  step_ += 1;
  time_ += dt_large_;

  // Print to screen
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n\n"
              << "TIME:  " << std::scientific << std::setprecision(12) << time_ << "/"
              << std::setprecision(4) << timemax_ << "     DT = " << std::scientific << dt_large_
              << "     STEP = " << std::setw(4) << step_ << "/" << std::setw(4) << numstep_ << "\n";
  }

  // prepare structure scatra field
  scatravec_[1]->ScaTraField()->PrepareTimeStep();
}

/*----------------------------------------------------------------------*
 | OuterLoop for sequentially staggered FS3I scheme          Thon 08/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScaleOuterLoop()
{
  DoStructScatraStep();

  if (DoesGrowthNeedsUpdate())  // includes the check for fsineedsupdate_
  {
    LargeTimeScaleDoGrowthUpdate();
  }
}

/*----------------------------------------------------------------------*
 | Do a large time scale structe scatra step                 Thon 08/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::DoStructScatraStep()
{
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n************************************************************************"
                 "\n                       AC STRUCTURE SCATRA SOLVER"
                 "\n************************************************************************\n"
              << std::endl;

    std::cout << "+- step/max -+-- scal-res/ abs-tol [norm] -+-- scal-inc/ rel-tol [norm] -+"
              << std::endl;
  }

  bool stopnonliniter = false;
  int itnum = 0;

  while (stopnonliniter == false)
  {
    StructScatraEvaluateSolveIterUpdate();
    itnum++;
    if (StructScatraConvergenceCheck(itnum)) break;
  }
}

/*--------------------------------------------------------------------------------*
 | evaluate, solver and iteratively update structure scalar problem    Thon 08/15 |
 *--------------------------------------------------------------------------------*/
void FS3I::ACFSI::StructScatraEvaluateSolveIterUpdate()
{
  if (infperm_) dserror("This not a valid option!");  // just for safety

  const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> scatra =
      scatravec_[1]->ScaTraField();  // structure scatra

  //----------------------------------------------------------------------
  // evaluate the structure scatra field
  //----------------------------------------------------------------------
  scatra->PrepareLinearSolve();

  //----------------------------------------------------------------------
  // calculate contributions due to finite interface permeability
  //----------------------------------------------------------------------
  EvaluateithScatraSurfacePermeability(1);

  //----------------------------------------------------------------------
  // recalculate fluid scatra contributions due to possible changed time step size
  // and the using of mean wss and mean phi for the fluid scatra field
  //----------------------------------------------------------------------
  EvaluateithScatraSurfacePermeability(0);

  //----------------------------------------------------------------------
  // add coupling to the resiudal
  //----------------------------------------------------------------------
  const Teuchos::RCP<Epetra_Vector> rhs_struct_scal = scatracoupforce_[1];
  const Teuchos::RCP<CORE::LINALG::SparseMatrix> mat_struct_scal = scatracoupmat_[1];
  const Teuchos::RCP<Epetra_Vector> residual = scatra->Residual();

  residual->Update(1.0, *rhs_struct_scal, 1.0);

  // add contribution of the fluid field
  Teuchos::RCP<Epetra_Vector> rhs_fluid_scal_boundary =
      scatrafieldexvec_[0]->ExtractVector(scatracoupforce_[0], 1);
  Teuchos::RCP<Epetra_Vector> rhs_fluid_scal =
      scatrafieldexvec_[1]->InsertVector(Scatra1ToScatra2(rhs_fluid_scal_boundary), 1);

  residual->Update(-1.0, *rhs_fluid_scal, 1.0);

  //----------------------------------------------------------------------
  // add coupling to the sysmat
  //----------------------------------------------------------------------
  const Teuchos::RCP<CORE::LINALG::SparseMatrix> sysmat = scatra->SystemMatrix();
  sysmat->Add(*mat_struct_scal, false, 1.0, 1.0);

  //----------------------------------------------------------------------
  // solve the scatra problem
  //----------------------------------------------------------------------
  const Teuchos::RCP<Epetra_Vector> structurescatraincrement =
      CORE::LINALG::CreateVector(*scatra->DofRowMap(), true);

  CORE::LINALG::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  scatra->Solver()->Solve(
      sysmat->EpetraOperator(), structurescatraincrement, residual, solver_params);

  //----------------------------------------------------------------------
  // update the strucutre scatra increment
  //----------------------------------------------------------------------
  scatra->UpdateIter(structurescatraincrement);
}

/*----------------------------------------------------------------------*
 | check convergence of structure scatra field               Thon 08/15 |
 *----------------------------------------------------------------------*/
bool FS3I::ACFSI::StructScatraConvergenceCheck(const int itnum)
{
  const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> scatra =
      scatravec_[1]->ScaTraField();  // structure scatra

  // some input parameters for the scatra fields
  const Teuchos::ParameterList& scatradyn =
      GLOBAL::Problem::Instance()->ScalarTransportDynamicParams();
  const int scatraitemax = scatradyn.sublist("NONLINEAR").get<int>("ITEMAX");
  const double scatraittol = scatradyn.sublist("NONLINEAR").get<double>("CONVTOL");
  const double scatraabstolres = scatradyn.sublist("NONLINEAR").get<double>("ABSTOLRES");


  double conresnorm(0.0);
  scatra->Residual()->Norm2(&conresnorm);
  double incconnorm(0.0);
  scatra->Increment()->Norm2(&incconnorm);
  double phinpnorm(0.0);
  scatra->Phinp()->Norm2(&phinpnorm);

  // care for the case that nothing really happens in the concentration field
  if (phinpnorm < 1e-5) phinpnorm = 1.0;

  // print the screen info
  if (Comm().MyPID() == 0)
  {
    printf("|   %3d/%3d  |  %1.3E/ %1.1E [L_2 ]  |  %1.3E/ %1.1E [L_2 ]  |\n", itnum, scatraitemax,
        conresnorm, scatraabstolres, incconnorm / phinpnorm, scatraittol);
  }

  // this is the convergence check
  // We always require at least one solve. We test the L_2-norm of the
  // current residual. Norm of residual is just printed for information
  if (conresnorm <= scatraabstolres and incconnorm / phinpnorm <= scatraittol)
  {
    if (Comm().MyPID() == 0)
    {
      // print 'finish line'
      printf("+------------+-----------------------------+-----------------------------+\n\n");
    }
    return true;
  }
  // if itemax is reached without convergence stop the simulation
  else if (itnum == scatraitemax)
  {
    if (Comm().MyPID() == 0)
    {
      printf("+---------------------------------------------------------------+\n");
      printf("|    scalar-scalar field did not converge in itemax steps!     |\n");
      printf("+---------------------------------------------------------------+\n");
    }
    // yes, we stop!
    //    dserror("Structure scatra not converged in itemax steps!");
    return true;
  }
  else
    return false;
}

/*----------------------------------------------------------------------*
 | Do we need to update the structure scatra displacments               |
 | due to growth                                             Thon 08/15 |
 *----------------------------------------------------------------------*/
bool FS3I::ACFSI::DoesGrowthNeedsUpdate()
{
  bool growthneedsupdate = false;

  // check if the structure material is a growth material. We assume here
  // that the structure has the same material for the whole discretiazation.
  // Hence we check only the first element:
  Teuchos::RCP<DRT::Discretization> structuredis = fsi_->StructureField()->Discretization();
  const int GID = structuredis->ElementColMap()->GID(0);  // global element ID

  Teuchos::RCP<MAT::Material> structurematerial = structuredis->gElement(GID)->Material();

  if (structurematerial->MaterialType() != INPAR::MAT::m_growth_volumetric)
  {
    dserror("In AC-FS3I we want growth, so use a growth material like MAT_GrowthVolumetric!");
  }
  else
  {
    //----------------------------------------------------------------------------------------------------
    // get alpha and growth inducing scalar
    //----------------------------------------------------------------------------------------------------
    double alpha = 0.0;
    int sc1 = 1;

    Teuchos::RCP<MAT::GrowthVolumetric> growthmaterial =
        Teuchos::rcp_dynamic_cast<MAT::GrowthVolumetric>(structurematerial);

    if (growthmaterial == Teuchos::null) dserror("Dynamic cast to MAT::GrowthVolumetric failed!");

    Teuchos::RCP<MAT::GrowthLaw> growthlaw = growthmaterial->Parameter()->growthlaw_;

    switch (growthlaw->MaterialType())
    {
      case INPAR::MAT::m_growth_ac:
      {
        Teuchos::RCP<MAT::GrowthLawAC> growthlawac =
            Teuchos::rcp_dynamic_cast<MAT::GrowthLawAC>(growthlaw);
        if (growthmaterial == Teuchos::null) dserror("Dynamic cast to MAT::GrowthLawAC failed!");
        alpha = growthlawac->Parameter()->alpha_;
        sc1 = growthlawac->Parameter()->Sc1_;
        break;
      }
      case INPAR::MAT::m_growth_ac_radial:
      {
        Teuchos::RCP<MAT::GrowthLawACRadial> growthlawacradial =
            Teuchos::rcp_dynamic_cast<MAT::GrowthLawACRadial>(growthlaw);
        if (growthlawacradial == Teuchos::null)
          dserror("Dynamic cast to MAT::GrowthLawACRadial failed!");
        alpha = growthlawacradial->Parameter()->alpha_;
        sc1 = growthlawacradial->Parameter()->Sc1_;
        break;
      }
      case INPAR::MAT::m_growth_ac_radial_refconc:
      {
        Teuchos::RCP<MAT::GrowthLawACRadialRefConc> growthlawacradialrefconc =
            Teuchos::rcp_dynamic_cast<MAT::GrowthLawACRadialRefConc>(growthlaw);
        if (growthlawacradialrefconc == Teuchos::null)
          dserror("Dynamic cast to MAT::GrowthLawACRadialRefConc failed!");
        alpha = growthlawacradialrefconc->Parameter()->alpha_;
        sc1 = growthlawacradialrefconc->Parameter()->Sc1_;
        break;
      }
      default:
      {
        dserror("Growth law not supported in AC-FS3I!");
        break;
      }
    }
    // Puh! That was exhausting. But we have to keep going.

    //----------------------------------------------------------------------------------------------------
    // get the approx. increase of volume due to growth since the beginning of the large time scale
    // loop
    //----------------------------------------------------------------------------------------------------
    const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> scatra =
        scatravec_[1]->ScaTraField();                                 // structure scatra
    const Teuchos::RCP<const Epetra_Vector> phinp = scatra->Phinp();  // fluidscatra

    // build difference vector with the reference
    const Teuchos::RCP<Epetra_Vector> phidiff_bltsl_ =
        CORE::LINALG::CreateVector(*scatra->DofRowMap(), true);
    phidiff_bltsl_->Update(1.0, *phinp, -1.0, *structurephinp_blts_, 0.0);

    // Extract the dof of interest
    Teuchos::RCP<Epetra_Vector> phidiff_bltsl_j =
        extractjthstructscalar_[sc1 - 1]->ExtractCondVector(phidiff_bltsl_);

    // get the maximum
    double max_phidiff_bltsl = 0.0;
    phidiff_bltsl_j->MaxValue(&max_phidiff_bltsl);

    //----------------------------------------------------------------------------------------------------
    // screen output
    //----------------------------------------------------------------------------------------------------
    const int growth_updates =
        GLOBAL::Problem::Instance()->FS3IDynamicParams().sublist("AC").get<int>("GROWTH_UPDATES");
    const double fsi_update_tol =
        GLOBAL::Problem::Instance()->FS3IDynamicParams().sublist("AC").get<double>(
            "FSI_UPDATE_TOL");

    if (Comm().MyPID() == 0)
      std::cout << std::scientific << std::setprecision(3)
                << "The maximal relative local growth since the small time scale is "
                << alpha * max_phidiff_bltsl << " (tol "
                << ((double)growth_updates_counter_ + 1.0) / (double)growth_updates * fsi_update_tol
                << ", iter " << growth_updates_counter_ << "/" << growth_updates << ")"
                << std::endl;

    // some safety check
    if (growth_updates_counter_ > growth_updates)
      dserror("It should not be possible to have done so much growth updates. Sorry!");

    //----------------------------------------------------------------------------------------------------
    // now the actual comparison
    //----------------------------------------------------------------------------------------------------
    // do we need a growth update?
    if (max_phidiff_bltsl * alpha >=
        ((double)growth_updates_counter_ + 1.0) / (double)growth_updates * fsi_update_tol)
    {
      growthneedsupdate = true;
    }

    // are we done with the current large time scale loop?
    if (max_phidiff_bltsl * alpha >= fsi_update_tol)
    {
      fsineedsupdate_ = true;
    }
  }

  return growthneedsupdate;
}

/*-------------------------------------------------------------------------*
 | update the structure scatra displacments due to growth       Thon 08/15 |
 *-------------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScaleDoGrowthUpdate()
{
  const int growth_updates =
      GLOBAL::Problem::Instance()->FS3IDynamicParams().sublist("AC").get<int>("GROWTH_UPDATES");

  const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> fluidscatra = scatravec_[0]->ScaTraField();
  const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> structurescatra = scatravec_[1]->ScaTraField();

  // Note: we never do never proceed with time_ and step_, so this really just about updating the
  // growth, i.e. the displacements of the structure scatra fields

  //----------------------------------------------------------------------
  // print to screen
  //----------------------------------------------------------------------
  if (Comm().MyPID() == 0)
  {
    std::cout << "\n************************************************************************"
                 "\n                         AC GROWTH UPDATE "
              << growth_updates_counter_ + 1 << "/" << growth_updates
              << "\n************************************************************************"
              << std::endl;
  }


  //----------------------------------------------------------------------
  // finish present structure scatra time step (no output)
  //----------------------------------------------------------------------
  structurescatra->Update();

  //----------------------------------------------------------------------
  // Switch time step of scatra fields
  //----------------------------------------------------------------------
  // Switch back the time step to do the update with the same (small) timestep as the fsi
  // (subcycling time step possible!)
  fluidscatra->SetDt(dt_);
  structurescatra->SetDt(dt_);

  //----------------------------------------------------------------------
  // Fix time_ and step_ counters
  //----------------------------------------------------------------------
  // time_+=dt_;

  SetTimeAndStepInFSI(time_ - dt_, step_ - 1);
  fluidscatra->SetTimeStep(time_ - dt_, step_ - 1);
  structurescatra->SetTimeStep(time_ - dt_, step_ - 1);

  // we now have to fix the time_ and step_ of the structure field, since this is not shifted
  // in PrepareTimeStep(), but in Update(), which we here will not call. So..
  fsi_->StructureField()->SetTime(time_ - dt_);
  fsi_->StructureField()->SetTimen(time_);
  fsi_->StructureField()->SetStep(step_ - 1);
  fsi_->StructureField()->SetStepn(step_);

  //----------------------------------------------------------------------
  // Prepare time steps
  //----------------------------------------------------------------------
  // fsi problem
  SetStructScatraSolution();
  fsi_->PrepareTimeStep();
  // scatra fields
  fluidscatra->PrepareTimeStep();
  structurescatra->PrepareTimeStep();

  //----------------------------------------------------------------------
  // do the growth update
  //----------------------------------------------------------------------
  // Safety check:
  CheckIfTimesAndStepsAndDtsMatch();

  // the actual calculations
  LargeTimeScaleOuterLoopIterStagg();

  //----------------------------------------------------------------------
  // write the output
  //----------------------------------------------------------------------
  // write fsi output. Scatra outputs are done later
  // fsi output
  constexpr bool force_prepare = false;
  fsi_->PrepareOutput(force_prepare);
  // NOTE: we have to call this functions, otherwise the structure displacements are not applied
  fsi_->Update();
  FsiOutput();
  // fluid scatra update. Structure scatra is done later
  fluidscatra->Update();
  fluidscatra->CheckAndWriteOutputAndRestart();

  //----------------------------------------------------------------------
  // Switch back time steps and set mean values in scatra fields
  //----------------------------------------------------------------------
  // Now set the time step back:
  fluidscatra->SetDt(dt_large_);
  structurescatra->SetDt(dt_large_);

  // set mean values in scatra fields
  LargeTimeScaleSetFSISolution();

  //----------------------------------------------------------------------
  // higher growth counter
  //----------------------------------------------------------------------
  growth_updates_counter_++;
}

/*-------------------------------------------------------------------------------*
 | OuterLoop for large time scale iterative staggered FS3I scheme     Thon 11/15 |
 *-------------------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScaleOuterLoopIterStagg()
{
  int itnum = 0;

  bool stopnonliniter = false;

  if (Comm().MyPID() == 0)
  {
    std::cout << "\n************************************************************************\n"
                 "                         OUTER ITERATION START"
              << "\n************************************************************************"
              << std::endl;
  }

  while (stopnonliniter == false)
  {
    itnum++;

    structureincrement_->Update(1.0, *fsi_->StructureField()->Dispnp(), 0.0);
    fluidincrement_->Update(1.0, *fsi_->FluidField()->Velnp(), 0.0);
    aleincrement_->Update(1.0, *fsi_->AleField()->Dispnp(), 0.0);

    SetStructScatraSolution();

    DoFSIStepStandard();
    // subcycling is not allowed, since we use this function for the growth update. Nevertheless it
    // should work.. periodical repetition is not allowed, since we want to converge the problems

    LargeTimeScaleSetFSISolution();

    SmallTimeScaleDoScatraStep();

    stopnonliniter = PartFs3iConvergenceCkeck(itnum);
  }
}

/*-----------------------------------------------------------------------*
 | set mean FSI values in scatra fields                       Thon 11/15 |
 *---------------------------------------------------- ------------------*/
void FS3I::ACFSI::LargeTimeScaleSetFSISolution()
{
  // we clear every state, including the states of the secondary dof sets
  for (unsigned i = 0; i < scatravec_.size(); ++i)
  {
    scatravec_[i]->ScaTraField()->Discretization()->ClearState(true);
    // we have to manually clear this since this can not be saved directly in the
    // primary dof set (because it is cleared in between)
    scatravec_[i]->ScaTraField()->ClearExternalConcentrations();
  }

  SetMeshDisp();
  SetMeanWallShearStresses();
  SetMeanFluidScatraConcentration();
  SetMembraneConcentration();
  // Set zeros velocities since we assume that the large time scale can not see the deformation of
  // the small time scale
  SetZeroVelocityField();
}

/*----------------------------------------------------------------------*
 | Update and output the large time scale                    Thon 08/15 |
 *----------------------------------------------------------------------*/
void FS3I::ACFSI::LargeTimeScaleUpdateAndOutput()
{
  // keep fsi time and fluid scatra field up to date
  SetTimeAndStepInFSI(time_, step_);
  scatravec_[0]->ScaTraField()->SetTimeStep(time_, step_);

  // NOTE: fsi output is already updated and written in LargeTimeScaleDoGrowthUpdate()
  // NOTE: fluid scatra is already updated and written in LargeTimeScaleDoGrowthUpdate()

  // now update and output the structure scatra field
  scatravec_[1]->ScaTraField()->Update();
  scatravec_[1]->ScaTraField()->CheckAndWriteOutputAndRestart();
}

/*----------------------------------------------------------------------*
 | Build map extractor which extracts the j-th dof           Thon 08/15 |
 *----------------------------------------------------------------------*/
std::vector<Teuchos::RCP<CORE::LINALG::MapExtractor>> FS3I::ACFSI::BuildMapExtractor()
{
  std::vector<Teuchos::RCP<CORE::LINALG::MapExtractor>> extractjthscalar;

  const Teuchos::RCP<SCATRA::ScaTraTimIntImpl> scatra =
      scatravec_[1]->ScaTraField();  // structure scatra
  const int numscal = scatra->NumScal();
  const Teuchos::RCP<const DRT::Discretization> dis = scatra->Discretization();

  for (int k = 0; k < numscal; k++)
  {
    std::set<int> conddofset;
    std::set<int> otherdofset;

    int numrownodes = dis->NumMyRowNodes();
    for (int i = 0; i < numrownodes; ++i)
    {
      DRT::Node* node = dis->lRowNode(i);

      std::vector<int> dof = dis->Dof(0, node);
      if (dof.size() != (unsigned)scatravec_[1]->ScaTraField()->NumScal())
        dserror("There was some error building the Map Extractor!");
      for (unsigned j = 0; j < dof.size(); ++j)
      {
        // test for dof position
        if (j != static_cast<unsigned>(k))
        {
          otherdofset.insert(dof[j]);
        }
        else
        {
          conddofset.insert(dof[j]);
        }
      }
    }
    std::vector<int> conddofmapvec;
    conddofmapvec.reserve(conddofset.size());
    conddofmapvec.assign(conddofset.begin(), conddofset.end());
    conddofset.clear();
    Teuchos::RCP<Epetra_Map> conddofmap = Teuchos::rcp(
        new Epetra_Map(-1, conddofmapvec.size(), conddofmapvec.data(), 0, dis->Comm()));
    conddofmapvec.clear();

    std::vector<int> otherdofmapvec;
    otherdofmapvec.reserve(otherdofset.size());
    otherdofmapvec.assign(otherdofset.begin(), otherdofset.end());
    otherdofset.clear();
    Teuchos::RCP<Epetra_Map> otherdofmap = Teuchos::rcp(
        new Epetra_Map(-1, otherdofmapvec.size(), otherdofmapvec.data(), 0, dis->Comm()));
    otherdofmapvec.clear();

    Teuchos::RCP<CORE::LINALG::MapExtractor> getjdof = Teuchos::rcp(new CORE::LINALG::MapExtractor);
    getjdof->Setup(*dis->DofRowMap(), conddofmap, otherdofmap);
    extractjthscalar.push_back(getjdof);
  }

  return extractjthscalar;
}

/*----------------------------------------------------------------------*
 | Compare if two doubles are relatively equal               Thon 08/15 |
 *----------------------------------------------------------------------*/
bool FS3I::ACFSI::IsRealtiveEqualTo(const double A, const double B, const double Ref)
{
  return ((fabs(A - B) / Ref) < 1e-12);
}

/*----------------------------------------------------------------------*
 | Compare if A mod B is relatively equal to zero            Thon 08/15 |
 *----------------------------------------------------------------------*/
bool FS3I::ACFSI::ModuloIsRealtiveZero(const double value, const double modulo, const double Ref)
{
  return IsRealtiveEqualTo(fmod(value + modulo / 2, modulo) - modulo / 2, 0.0, Ref);
}

/*----------------------------------------------------------------------*
 | Compare if A mod B is relatively equal to zero            Thon 10/15 |
 *----------------------------------------------------------------------*/
FS3I::MeanManager::MeanManager(
    const Epetra_Map& wssmap, const Epetra_Map& phimap, const Epetra_Map& pressuremap)
    : SumWss_(CORE::LINALG::CreateVector(wssmap, true)),
      SumPhi_(CORE::LINALG::CreateVector(phimap, true)),
      SumPres_(CORE::LINALG::CreateVector(pressuremap, true)),
      SumDtWss_(0.0),
      SumDtPhi_(0.0),
      SumDtPres_(0.0)
{
}


/*----------------------------------------------------------------------*
 | add value into the mean manager                           Thon 10/15 |
 *----------------------------------------------------------------------*/
void FS3I::MeanManager::AddValue(
    const std::string type, const Teuchos::RCP<const Epetra_Vector> value, const double dt)
{
  if (type == "wss")
  {
#ifdef BACI_DEBUG
    // check, whether maps are the same
    if (not value->Map().PointSameAs(SumWss_->Map()))
    {
      dserror("Maps do not match, but they have to.");
    }
#endif

    SumWss_->Update(dt, *value, 1.0);  // weighted sum of all prior stresses
    SumDtWss_ += dt;
  }
  else if (type == "phi")
  {
#ifdef BACI_DEBUG
    // check, whether maps are the same
    if (not value->Map().PointSameAs(SumPhi_->Map()))
    {
      dserror("Maps do not match, but they have to.");
    }
#endif

    SumPhi_->Update(dt, *value, 1.0);  // weighted sum of all prior stresses
    SumDtPhi_ += dt;
  }
  else if (type == "pressure")
  {
#ifdef BACI_DEBUG
    // check, whether maps are the same
    if (not value->Map().PointSameAs(SumPres_->Map()))
    {
      dserror("Maps do not match, but they have to.");
    }
#endif

    SumPres_->Update(dt, *value, 1.0);  // weighted sum of all prior stresses
    SumDtPres_ += dt;
  }
  else
    dserror("Mean Manager does not support the given value '%s'.", type.c_str());

  return;
}

/*----------------------------------------------------------------------*
 | reset mean manager                                        Thon 10/15 |
 *----------------------------------------------------------------------*/
void FS3I::MeanManager::Reset()
{
  // first some checking
  if (abs(SumDtWss_ - SumDtPhi_) > 1e-14 or abs(SumDtWss_ - SumDtPres_) > 1e-14)
    dserror("The time ranges you did mean over do not match!");

  SumWss_->PutScalar(0.0);
  SumDtWss_ = 0.0;
  SumPhi_->PutScalar(0.0);
  SumDtPhi_ = 0.0;
  SumPres_->PutScalar(0.0);
  SumDtPres_ = 0.0;
}

/*----------------------------------------------------------------------*
 | get some mean value                                       Thon 10/15 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<const Epetra_Vector> FS3I::MeanManager::GetMeanValue(const std::string type) const
{
  Teuchos::RCP<Epetra_Vector> meanvector;

  if (type == "mean_wss")
  {
    meanvector = Teuchos::rcp(new Epetra_Vector(SumWss_->Map(), true));

    if (SumDtWss_ > 1e-12)  // iff we have actually calculated some mean wss
      meanvector->Update(1.0 / SumDtWss_, *SumWss_, 0.0);  // weighted sum of all prior stresses
    else
    {
      double norm = 0.0;
      meanvector->NormInf(&norm);
      if (norm > 1e-12)
        dserror("SumDtWss_ is zero, but SumWss_ not.. Something is terribly wrong!");
    }
  }
  else if (type == "osi")
  {
    dserror("Oscillatory shear index is yet not supported!");
  }
  else if (type == "mean_phi")
  {
    meanvector = Teuchos::rcp(new Epetra_Vector(SumPhi_->Map(), true));

    if (SumDtPhi_ > 1e-12)  // iff we have actually calculated some mean wss
      meanvector->Update(1.0 / SumDtPhi_, *SumPhi_, 0.0);  // weighted sum of all prior stresses
    else
    {
      double norm = 0.0;
      meanvector->NormInf(&norm);
      if (norm > 1e-12)
        dserror("SumDtPhi_ is zero, but SumPhi_ not.. Something is terribly wrong!");
    }
  }
  else if (type == "mean_pressure")
  {
    meanvector = Teuchos::rcp(new Epetra_Vector(SumPres_->Map(), true));

    if (SumDtPres_ > 1e-12)  // iff we have actually calculated some mean wss
      meanvector->Update(1.0 / SumDtPres_, *SumPres_, 0.0);  // weighted sum of all prior stresses
    else
    {
      double norm = 0.0;
      meanvector->NormInf(&norm);
      if (norm > 1e-12)
        dserror("SumDtPres_ is zero, but SumPres_ not.. Something is terribly wrong!");
    }
  }
  else
    dserror("Mean Manager does not support the given value '%s'.", type.c_str());

  return meanvector;
}

/*----------------------------------------------------------------------*
 | Write restart of mean manager                             Thon 10/15 |
 *----------------------------------------------------------------------*/
void FS3I::MeanManager::WriteRestart(Teuchos::RCP<IO::DiscretizationWriter> fluidwriter) const
{
  // first some checking
  if (abs(SumDtWss_ - SumDtPhi_) > 1e-14 or abs(SumDtWss_ - SumDtPres_) > 1e-14)
    dserror("The time ranges you did mean over do not match!");

  // write all values
  fluidwriter->WriteVector("SumWss", SumWss_);
  fluidwriter->WriteVector("SumPhi", SumPhi_);
  //  fluidwriter->WriteVector("SumPres", SumPres_);
  // we need only one SumDt since they are all the same
  fluidwriter->WriteDouble("SumDtWss", SumDtWss_);
}

/*----------------------------------------------------------------------*
 | Read restart of mean manager                             Thon 10/15 |
 *----------------------------------------------------------------------*/
void FS3I::MeanManager::ReadRestart(IO::DiscretizationReader& fluidreader)
{
  // read all values...
  fluidreader.ReadVector(SumWss_, "SumWss");
  fluidreader.ReadVector(SumPhi_, "SumPhi");
  //  fluidreader.ReadVector(SumPres_, "SumPres");
  SumDtWss_ = fluidreader.ReadDouble("SumDtWss");
  //...and recover the rest
  SumDtPhi_ = SumDtWss_;
  SumDtPres_ = SumDtWss_;
}

BACI_NAMESPACE_CLOSE
