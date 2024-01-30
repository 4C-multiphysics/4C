/*----------------------------------------------------------------------*/
/*! \file

\brief Abstract class to be overloaded by different constraint enforcement techniques for fluid-beam
interaction.

\level 3

*----------------------------------------------------------------------*/

#include "baci_fbi_constraintenforcer.H"

#include "baci_adapter_fld_fbi_movingboundary.H"
#include "baci_adapter_str_fbiwrapper.H"
#include "baci_beaminteraction_calc_utils.H"  // todo put this into bridge to keep everything beam specific in there
#include "baci_beaminteraction_contact_pair.H"
#include "baci_binstrategy.H"
#include "baci_fbi_adapter_constraintbridge.H"
#include "baci_fbi_adapter_constraintbridge_penalty.H"
#include "baci_fbi_beam_to_fluid_meshtying_output_params.H"
#include "baci_fbi_beam_to_fluid_meshtying_params.H"
#include "baci_fbi_immersed_geometry_coupler.H"
#include "baci_fluid_utils.H"
#include "baci_geometry_pair.H"
#include "baci_global_data.H"
#include "baci_inpar_fbi.H"
#include "baci_inpar_fluid.H"
#include "baci_lib_discret.H"
#include "baci_lib_discret_faces.H"
#include "baci_lib_element.H"
#include "baci_lib_node.H"
#include "baci_lib_utils.H"
#include "baci_lib_utils_parallel.H"
#include "baci_linalg_blocksparsematrix.H"
#include "baci_linalg_fixedsizematrix.H"
#include "baci_linalg_mapextractor.H"
#include "baci_linalg_utils_sparse_algebra_create.H"

#include <iostream>

BACI_NAMESPACE_OPEN

ADAPTER::FBIConstraintenforcer::FBIConstraintenforcer(
    Teuchos::RCP<ADAPTER::FBIConstraintBridge> bridge,
    Teuchos::RCP<FBI::FBIGeometryCoupler> geometrycoupler)
    : fluid_(Teuchos::null),
      structure_(Teuchos::null),
      discretizations_(),
      bridge_(bridge),
      geometrycoupler_(geometrycoupler),
      column_structure_displacement_(Teuchos::null),
      column_structure_velocity_(Teuchos::null),
      column_fluid_velocity_(Teuchos::null),
      velocity_pressure_splitter_(Teuchos::rcp(new CORE::LINALG::MapExtractor()))
{
}

/*----------------------------------------------------------------------*/

void ADAPTER::FBIConstraintenforcer::Setup(Teuchos::RCP<ADAPTER::FSIStructureWrapper> structure,
    Teuchos::RCP<ADAPTER::FluidMovingBoundary> fluid)
{
  fluid_ = fluid;
  structure_ = structure;
  discretizations_.push_back(structure_->Discretization());
  discretizations_.push_back(fluid_->Discretization());

  CORE::LINALG::CreateMapExtractorFromDiscretization(
      *(fluid_->Discretization()), 3, *velocity_pressure_splitter_);

  bool meshtying =
      (GLOBAL::Problem::Instance()->FluidDynamicParams().get<std::string>("MESHTYING") != "no");

  Teuchos::RCP<CORE::LINALG::SparseOperator> fluidmatrix(Teuchos::null);

  if (meshtying)
  {
    if (structure_->Discretization()->Comm().NumProc() > 1)
      dserror(
          "Currently fluid mesh tying can only be used for serial computations, since offproc "
          "assembly is not supported. Once the coupling matrices are computed by the fluid element "
          "owner, this will change.");

    fluidmatrix = (Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->GetMeshtying())
                      ->InitSystemMatrix();
  }
  else
  {
    fluidmatrix = Teuchos::rcp(
        new CORE::LINALG::SparseMatrix(*(fluid_->Discretization()->DofRowMap()), 30, true, true,
            CORE::LINALG::SparseMatrix::FE_MATRIX));  // todo Is there a better estimator?
  }

  bridge_->Setup(structure_->Discretization()->DofRowMap(), fluid_->Discretization()->DofRowMap(),
      fluidmatrix, meshtying);
  if (structure_->Discretization()->Comm().NumProc() > 1)
  {
    geometrycoupler_->ExtendBeamGhosting(*(structure->Discretization()));

    // After ghosting we need to explicitly set up the MultiMapExtractor again
    Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(structure_, true)
        ->SetupMultiMapExtractor();
  }

  geometrycoupler_->Setup(discretizations_,
      DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Dispnp()));
}

/*----------------------------------------------------------------------*/

void ADAPTER::FBIConstraintenforcer::Evaluate()
{
  // We use the column vectors here, because currently the search is based on neighboring nodes,
  // but the element pairs are created using the elements needing all information on all their
  // DOFs
  column_structure_displacement_ =
      DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Dispnp());
  column_structure_velocity_ =
      DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Velnp());
  column_fluid_velocity_ = DRT::UTILS::GetColVersionOfRowVector(fluid_->Discretization(),
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->Velnp());

  geometrycoupler_->UpdateBinning(discretizations_[0], column_structure_displacement_);

  // Before each search we delete all pair and segment information
  bridge_->Clear();
  bridge_->ResetBridge();

  // Do the search in the geometrycoupler_ and return the possible pair ids
  Teuchos::RCP<std::map<int, std::vector<int>>> pairids = geometrycoupler_->Search(discretizations_,
      column_structure_displacement_);  // todo make this a vector? At some point we probably
                                        // need the ale displacements as well

  // For now we need to separate the pair creation from the search, since the search takes place
  // on the fluid elements owner, while (for now) the pair has to be created on the beam element
  // owner
  CreatePairs(pairids);

  // Create all needed matrix and vector contributions based on the current state
  bridge_->Evaluate(discretizations_[0], discretizations_[1],
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->Velnp(), structure_->Velnp());
}

/*----------------------------------------------------------------------*/

Teuchos::RCP<Epetra_Vector> ADAPTER::FBIConstraintenforcer::StructureToFluid(int step)
{
  // todo only access the parameter list once

  // Check if we want to couple the fluid
  const Teuchos::ParameterList& fbi = GLOBAL::Problem::Instance()->FBIParams();
  if (Teuchos::getIntegralValue<INPAR::FBI::BeamToFluidCoupling>(fbi, "COUPLING") !=
          INPAR::FBI::BeamToFluidCoupling::solid &&
      fbi.get<int>("STARTSTEP") < step)
  {
    // Assemble the fluid stiffness matrix and hand it to the fluid solver
    Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)
        ->SetCouplingContributions(AssembleFluidCouplingMatrix());

    // Assemble the fluid force vector and hand it to the fluid solver
    fluid_->ApplyInterfaceValues(AssembleFluidCouplingResidual());
  }

  // return the current struture velocity
  return Teuchos::rcp_dynamic_cast<ADAPTER::FBIStructureWrapper>(structure_, true)
      ->ExtractInterfaceVelnp();
};

/*----------------------------------------------------------------------*/
void ADAPTER::FBIConstraintenforcer::RecomputeCouplingWithoutPairCreation()
{
  // Before each search we delete all pair and segment information
  bridge_->ResetBridge();

  ResetAllPairStates();

  // Create all needed matrix and vector contributions based on the current state
  bridge_->Evaluate(discretizations_[0], discretizations_[1],
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->Velnp(), structure_->Velnp());
};

/*----------------------------------------------------------------------*/
// return the structure force
Teuchos::RCP<Epetra_Vector> ADAPTER::FBIConstraintenforcer::FluidToStructure()
{
  return AssembleStructureCouplingResidual();
};

/*----------------------------------------------------------------------*/

// For now we need to separate the pair creation from the search, since the search takes place on
// the fluid elements owner, while (for now) the pair has to be created on the beam element owner
void ADAPTER::FBIConstraintenforcer::CreatePairs(
    Teuchos::RCP<std::map<int, std::vector<int>>> pairids)
{
  if ((structure_->Discretization())->Comm().NumProc() > 1)
  {
    // The geometrycoupler takes care of all MPI communication that needs to be done before the
    // pairs can finally be created
    geometrycoupler_->PreparePairCreation(discretizations_, pairids);

    column_structure_displacement_ =
        DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Dispnp());
    column_structure_velocity_ =
        DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Velnp());
    column_fluid_velocity_ = DRT::UTILS::GetColVersionOfRowVector(fluid_->Discretization(),
        Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->Velnp());
  }


  std::vector<DRT::Element const*> ele_ptrs(2);
  std::vector<double> beam_dofvec = std::vector<double>();
  std::vector<double> fluid_dofvec = std::vector<double>();

  // loop over all (embedded) beam elements
  std::map<int, std::vector<int>>::const_iterator beamelementiterator;
  for (beamelementiterator = pairids->begin(); beamelementiterator != pairids->end();
       beamelementiterator++)
  {
    // add beam elements to the element pair pointer
    ele_ptrs[0] = (structure_->Discretization())->gElement(beamelementiterator->first);


    if (ele_ptrs[0]->Owner() != structure_->Discretization()->Comm().MyPID())
      dserror(
          "For now we can only create the pair on the beam owner, but beam element owner is %i "
          "and "
          "we are on proc %i \n",
          ele_ptrs[0]->Owner(), structure_->Discretization()->Comm().MyPID());

    // loop over all fluid elements, in which the beam element might lie
    for (std::vector<int>::const_iterator fluideleIter = beamelementiterator->second.begin();
         fluideleIter != (beamelementiterator->second).end(); fluideleIter++)
    {
      DRT::Element* fluidele = (fluid_->Discretization())->gElement(*fluideleIter);

      // add fluid element to the element pair pointer
      ele_ptrs[1] = fluidele;

      // Extract current element dofs, i.e. positions and velocities
      ExtractCurrentElementDofs(ele_ptrs, beam_dofvec, fluid_dofvec);

      // Finally tell the bridge to create the pair
      bridge_->CreatePair(ele_ptrs, beam_dofvec, fluid_dofvec);
    }
  }
}
/*----------------------------------------------------------------------*/
void ADAPTER::FBIConstraintenforcer::ResetAllPairStates()
{
  // Get current state
  column_structure_displacement_ =
      DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Dispnp());
  column_structure_velocity_ =
      DRT::UTILS::GetColVersionOfRowVector(structure_->Discretization(), structure_->Velnp());
  column_fluid_velocity_ = DRT::UTILS::GetColVersionOfRowVector(fluid_->Discretization(),
      Teuchos::rcp_dynamic_cast<ADAPTER::FBIFluidMB>(fluid_, true)->Velnp());

  std::vector<DRT::Element const*> ele_ptrs(2);
  std::vector<double> beam_dofvec = std::vector<double>();
  std::vector<double> fluid_dofvec = std::vector<double>();

  for (auto pairiterator = bridge_->GetPairs()->begin(); pairiterator != bridge_->GetPairs()->end();
       pairiterator++)
  {
    ele_ptrs[0] = (*pairiterator)->Element1();
    ele_ptrs[1] = (*pairiterator)->Element2();

    // Extract current element dofs, i.e. positions and velocities
    ExtractCurrentElementDofs(ele_ptrs, beam_dofvec, fluid_dofvec);

    // Finally tell the bridge to create the pair
    bridge_->ResetPair(beam_dofvec, fluid_dofvec, *pairiterator);
  }
}
/*----------------------------------------------------------------------*/

void ADAPTER::FBIConstraintenforcer::ExtractCurrentElementDofs(
    std::vector<DRT::Element const*> elements, std::vector<double>& beam_dofvec,
    std::vector<double>& fluid_dofvec) const
{
  std::vector<double> vel_tmp;

  // extract the current position of the beam element from the displacement vector
  BEAMINTERACTION::UTILS::ExtractPosDofVecAbsoluteValues(*(structure_->Discretization()),
      elements[0], column_structure_displacement_,
      beam_dofvec);  // todo get "interface" displacements only for beam
                     // elements
  // extract velocity of the beam element
  BEAMINTERACTION::UTILS::ExtractPosDofVecValues(
      *(structure_->Discretization()), elements[0], column_structure_velocity_, vel_tmp);

  for (double val : vel_tmp) beam_dofvec.push_back(val);

  vel_tmp.clear();
  // extract the current positions and velocities of the fluid element todo only valid for fixed
  // grid, not for ALE
  fluid_dofvec.clear();
  const DRT::Node* const* fluidnodes = elements[1]->Nodes();
  for (int lid = 0; lid < elements[1]->NumNode(); ++lid)
  {
    for (int dim = 0; dim < 3; dim++)
    {
      fluid_dofvec.push_back(fluidnodes[lid]->X()[dim]);
    }
  }

  // extract current fluid velocities
  BEAMINTERACTION::UTILS::GetCurrentElementDis(
      *(fluid_->Discretization()), elements[1], column_fluid_velocity_, vel_tmp);

  // todo This is a very crude way to separate the pressure from the velocity dofs.. maybe just
  // use an extractor?
  for (unsigned int i = 0; i < vel_tmp.size(); i++)
  {
    if ((i + 1) % 4) fluid_dofvec.push_back(vel_tmp[i]);
  }
}

/*----------------------------------------------------------------------*/

void ADAPTER::FBIConstraintenforcer::SetBinning(Teuchos::RCP<BINSTRATEGY::BinningStrategy> binning)
{
  geometrycoupler_->SetBinning(binning);
};

BACI_NAMESPACE_CLOSE
