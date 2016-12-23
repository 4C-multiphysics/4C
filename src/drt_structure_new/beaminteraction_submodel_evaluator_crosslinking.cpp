/*-----------------------------------------------------------*/
/*!
\file beaminteraction_submodel_evaluator_crosslinking.cpp

\brief class for submodel crosslinking

\maintainer Jonas Eichinger, Maximilian Grill

\level 3

*/
/*-----------------------------------------------------------*/


#include "beaminteraction_submodel_evaluator_crosslinking.H"
#include "str_model_evaluator_beaminteraction_datastate.H"
#include "str_timint_basedataglobalstate.H"
#include "str_utils.H"

#include "../drt_lib/drt_dserror.H"
#include "../drt_io/io.H"
#include "../drt_io/io_pstream.H"

#include <Epetra_Comm.h>
#include <Teuchos_TimeMonitor.hpp>
#include <fenv.h>

#include "../drt_particle/particle_handler.H"

#include "crosslinking_params.H"
#include "../drt_biopolynet/biopolynet_calc_utils.H"
#include "../drt_beamcontact/beam_to_beam_linkage.H"
#include "../drt_biopolynet/crosslinker_node.H"

#include <Epetra_FEVector.h>

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Crosslinking() :
    crosslinking_params_ptr_( Teuchos::null ),
    bin_beamcontent_( INPAR::BINSTRATEGY::Beam )
{
  crosslinker_data_.clear();
  beam_data_.clear();
  doublebondcl_.clear();
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Setup()
{
  CheckInit();

  // construct, init and setup data container for crosslinking
  crosslinking_params_ptr_ = Teuchos::rcp( new BEAMINTERACTION::CrosslinkingParams() );
  crosslinking_params_ptr_->Init();
  crosslinking_params_ptr_->Setup();

  // gather data for all column crosslinker and beams initially
  PreComputeCrosslinkerData();
  PreComputeBeamData();

  // set flag
  issetup_ = true;
}

/*-------------------------------------------------------------------------------*
 *-------------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Reset()
{
  CheckInitSetup();

  // reset crosslinker pairs
  std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::const_iterator iter;
  for ( iter = doublebondcl_.begin(); iter != doublebondcl_.end(); ++iter )
  {
    Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> elepairptr = iter->second;

    // init positions and triads
    std::vector<LINALG::Matrix<3,1> > pos(2);
    std::vector<LINALG::Matrix<3,3> > triad(2);

    for( int i = 0; i < 2; ++i )
    {
      int elegid = elepairptr->GetEleGid(i);
      int locbspotnum = elepairptr->GetLocBSpotNum(i);
      DRT::Element* ele = DiscretPtr()->gElement(elegid);

      BIOPOLYNET::UTILS::GetPosAndTriadOfBindingSpot( Discret(), ele,
          BeamInteractionDataStatePtr()->GetMutableDisColNp(),
          PeriodicBoundingBoxPtr(), locbspotnum, pos[i], triad[i] );
    }

    // unshift one of the positions if both are separated by a periodic boundary
    // condition, i.e. have been shifted before
    PeriodicBoundingBoxPtr()->UnShift3D( pos[1], pos[0] );

    // finally reset state
    elepairptr->ResetState( pos, triad );
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::EvaluateForce()
{
  CheckInitSetup();

  // force and moment exerted on the two connection sites due to the mechanical connection
  LINALG::SerialDenseVector bspotforce1(6);
  LINALG::SerialDenseVector bspotforce2(6);

  // resulting discrete element force vectors of the two parent elements
  LINALG::SerialDenseVector ele1force;
  LINALG::SerialDenseVector ele2force;

  Epetra_SerialDenseMatrix dummystiff(0,0);

  std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::const_iterator iter;
  for ( iter = doublebondcl_.begin(); iter != doublebondcl_.end(); ++iter )
  {
    Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> elepairptr = iter->second;

    // zero out variables
    bspotforce1.Zero();
    bspotforce2.Zero();

    // evaluate beam linkage object to get forces and moments on binding spots
    elepairptr->EvaluateForce(bspotforce1,bspotforce2);

    // apply forces on binding spots to parent elements
    // and get their discrete element force vectors
    BIOPOLYNET::UTILS::ApplyBpotForceStiffToParentElements(
        Discret(),
        BeamInteractionDataStatePtr()->GetMutableDisColNp(),
        elepairptr,
        bspotforce1,
        bspotforce2,
        dummystiff,
        dummystiff,
        dummystiff,
        dummystiff,
        &ele1force,
        &ele2force,
        NULL,
        NULL,
        NULL,
        NULL);

    // assemble the contributions into force vector class variable
    // f_crosslink_np_ptr_, i.e. in the DOFs of the connected nodes
    BIOPOLYNET::UTILS::FEAssembleEleForceStiffIntoSystemVectorMatrix(
        *DiscretPtr(),
        elepairptr->GetEleGid(0),
        elepairptr->GetEleGid(1),
        ele1force,
        ele2force,
        dummystiff,
        dummystiff,
        dummystiff,
        dummystiff,
        BeamInteractionDataStatePtr()->GetMutableForceNp(),
        Teuchos::null);
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::EvaluateStiff()
{
  CheckInitSetup();

  /* linearizations, i.e. stiffness contributions due to forces on the two
   * connection sites due to the mechanical connection */
  LINALG::SerialDenseMatrix bspotstiff11(6,6);
  LINALG::SerialDenseMatrix bspotstiff12(6,6);
  LINALG::SerialDenseMatrix bspotstiff21(6,6);
  LINALG::SerialDenseMatrix bspotstiff22(6,6);

  // linearizations, i.e. discrete stiffness contributions to the two parent elements
  // we can't handle this separately for both elements because there are entries which couple the two element stiffness blocks
  LINALG::SerialDenseMatrix ele11stiff;
  LINALG::SerialDenseMatrix ele12stiff;
  LINALG::SerialDenseMatrix ele21stiff;
  LINALG::SerialDenseMatrix ele22stiff;

  Epetra_SerialDenseVector dummyforce(0);

  std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::const_iterator iter;
  for ( iter = doublebondcl_.begin(); iter != doublebondcl_.end(); ++iter )
  {
    Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> elepairptr = iter->second;

    // zero out variables
    bspotstiff11.Zero();
    bspotstiff12.Zero();
    bspotstiff21.Zero();
    bspotstiff22.Zero();

     // evaluate beam linkage object to get linearizations of forces and moments on binding spots
    elepairptr->EvaluateStiff(
        bspotstiff11,
        bspotstiff12,
        bspotstiff21,
        bspotstiff22);

    // apply linearizations to parent elements and get their discrete element stiffness matrices
    BIOPOLYNET::UTILS::ApplyBpotForceStiffToParentElements(
        Discret(),
        BeamInteractionDataStatePtr()->GetMutableDisColNp(),
        elepairptr,
        dummyforce,
        dummyforce,
        bspotstiff11,
        bspotstiff12,
        bspotstiff21,
        bspotstiff22,
        NULL,
        NULL,
        &ele11stiff,
        &ele12stiff,
        &ele21stiff,
        &ele22stiff);

    // assemble the contributions into stiffness matrix class variable
    // stiff_crosslink_ptr_, i.e. in the DOFs of the connected nodes
    BIOPOLYNET::UTILS::FEAssembleEleForceStiffIntoSystemVectorMatrix(
        *DiscretPtr(),
        elepairptr->GetEleGid(0),
        elepairptr->GetEleGid(1),
        dummyforce,
        dummyforce,
        ele11stiff,
        ele12stiff,
        ele21stiff,
        ele22stiff,
        Teuchos::null,
        BeamInteractionDataStatePtr()->GetMutableStiff());
   }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::EvaluateForceStiff()
{
  CheckInitSetup();

  Teuchos::RCP<Epetra_FEVector> fe_sysvec =
      Teuchos::rcp(new Epetra_FEVector(*DiscretPtr()->DofRowMap()));

  // force and moment exerted on the two connection sites due to the mechanical connection
  LINALG::SerialDenseVector bspotforce1(6);
  LINALG::SerialDenseVector bspotforce2(6);

  /* linearizations, i.e. stiffness contributions due to forces on the two
   * connection sites due to the mechanical connection */
  LINALG::SerialDenseMatrix bspotstiff11(6,6);
  LINALG::SerialDenseMatrix bspotstiff12(6,6);
  LINALG::SerialDenseMatrix bspotstiff21(6,6);
  LINALG::SerialDenseMatrix bspotstiff22(6,6);

  // resulting discrete element force vectors of the two parent elements
  LINALG::SerialDenseVector ele1force;
  LINALG::SerialDenseVector ele2force;

  // linearizations, i.e. discrete stiffness contributions to the two parent elements
  // we can't handle this separately for both elements because there are entries which couple the two element stiffness blocks
  LINALG::SerialDenseMatrix ele11stiff;
  LINALG::SerialDenseMatrix ele12stiff;
  LINALG::SerialDenseMatrix ele21stiff;
  LINALG::SerialDenseMatrix ele22stiff;


  std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::const_iterator iter;
  for (iter=doublebondcl_.begin(); iter!=doublebondcl_.end(); ++iter)
  {
    Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> elepairptr = iter->second;

    // zero out variables
    bspotforce1.Zero();
    bspotforce2.Zero();
    bspotstiff11.Zero();
    bspotstiff12.Zero();
    bspotstiff21.Zero();
    bspotstiff22.Zero();

    // evaluate beam linkage object to get forces and moments on binding spots
    elepairptr->EvaluateForceStiff(bspotforce1,
                                   bspotforce2,
                                   bspotstiff11,
                                   bspotstiff12,
                                   bspotstiff21,
                                   bspotstiff22);

    // apply forces on binding spots and corresponding linearizations to parent elements
    // and get their discrete element force vectors and stiffness matrices
    BIOPOLYNET::UTILS::ApplyBpotForceStiffToParentElements(
        Discret(),
        BeamInteractionDataStatePtr()->GetMutableDisColNp(),
        elepairptr,
        bspotforce1,
        bspotforce2,
        bspotstiff11,
        bspotstiff12,
        bspotstiff21,
        bspotstiff22,
        &ele1force,
        &ele2force,
        &ele11stiff,
        &ele12stiff,
        &ele21stiff,
        &ele22stiff);

    // assemble the contributions into force and stiffness class variables
    // f_crosslink_np_ptr_, stiff_crosslink_ptr_, i.e. in the DOFs of the connected nodes
    BIOPOLYNET::UTILS::FEAssembleEleForceStiffIntoSystemVectorMatrix(
        *DiscretPtr(),
        elepairptr->GetEleGid(0),
        elepairptr->GetEleGid(1),
        ele1force,
        ele2force,
        ele11stiff,
        ele12stiff,
        ele21stiff,
        ele22stiff,
        BeamInteractionDataStatePtr()->GetMutableForceNp(),
        BeamInteractionDataStatePtr()->GetMutableStiff());
  }

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UpdateStepState(
    const double& timefac_n)
{
  CheckInitSetup();

}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PreUpdateStepElement()
{
  CheckInitSetup();

  // -------------------------------------------------------------------------
  // crosslinker diffusion:
  //    - according to browninan dyn for free cl
  //    - according to beams for single and double bonded
  // -------------------------------------------------------------------------
  // note: it is possible that a crosslinker leaves the computational domain
  // at this point, it gets shifted back in in UpdateBinStrategy which is called
  // right after this
  DiffuseCrosslinker();

  // erase temporary data container as both distributions as well the col data
  // might have changed
  crosslinker_data_.clear();

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UpdateStepElement()
{
  CheckInitSetup();

  TEUCHOS_FUNC_TIME_MONITOR("STR::MODELEVALUATOR::Crosslinking::UpdateCrosslinking");

  // -------------------------------------------------------------------------
  // update double bonded linker
  // -------------------------------------------------------------------------
  UpdateMyDoubleBondsAfterRedistribution();

  // -------------------------------------------------------------------------
  // now we manage binding events, this includes:
  //  - find potential binding events on each proc
  //  - make a decision by asking other procs
  //  - set bonds and adapt states accordingly
  // after that, we we manage unbinding events
  // -------------------------------------------------------------------------

  // intended bonds of row crosslinker on myrank (key is clgid)
  std::map<int, BindEventData > mybonds;
  // intended bond col crosslinker to row element (key is owner of crosslinker != myrank)
  std::map<int, std::vector<BindEventData> > undecidedbonds;

  // fill binding event maps
  FindPotentialBindingEvents( mybonds, undecidedbonds );

  // bind events where myrank only owns the elements, cl are taken care
  // of by their owner (key is clgid)
  std::map<int, BindEventData > myelebonds;

  // now each row owner of a linker gets requests, makes a random decision and
  // informs back its requesters
  ManageBindingInParallel( mybonds, undecidedbonds, myelebonds );

  // actual update of binding states is done here
  BindMyCrosslinkerAndElements( mybonds, myelebonds );

  // -------------------------------------------------------------------------
  // unbinding events if probability check is passed
  // -------------------------------------------------------------------------
  UnBindCrosslinker();

  beam_data_.clear();

#ifdef DEBUG
  // safety check if a crosslinker got lost
  if(BinDiscretPtr()->NumGlobalNodes()!=crosslinking_params_ptr_->NumCrosslink())
    dserror("A crosslinker got lost, something went wrong.");
#endif

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PostUpdateStepElement()
{
  CheckInitSetup();

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::OutputStepState(
    IO::DiscretizationWriter& iowriter) const
{
  CheckInitSetup();

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ResetStepState()
{
  CheckInitSetup();

  dserror("Not yet implemented");


}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::DiffuseCrosslinker()
{
  CheckInitSetup();

  // get standard deviation and mean value for crosslinker that are free to
  // diffuse
  double standarddev = std::sqrt(2.0 * crosslinking_params_ptr_->KT() /
                       (3.0*M_PI * crosslinking_params_ptr_->Viscosity()
                       * crosslinking_params_ptr_->LinkingLength())     // Todo check the scalar factor
                       * (*GState().GetDeltaTime())[0]);
  double meanvalue = 0.0;
  // Set mean value and standard deviation of normal distribution
  // FixMe standard deviation = sqrt(variance) check this for potential error !!!
  DRT::Problem::Instance()->Random()->SetMeanVariance(meanvalue,standarddev);

  // loop over all row crosslinker (beam binding status not touched here)
  const int numrowcl = BinDiscretPtr()->NumMyRowNodes();
  for(int rowcli = 0; rowcli < numrowcl; ++rowcli )
  {
    // get current linker
    CROSSLINKING::CrosslinkerNode* crosslinker_i =
        dynamic_cast<CROSSLINKING::CrosslinkerNode*>(BinDiscretPtr()->lRowNode(rowcli));

#ifdef DEBUG
      if(crosslinker_i == NULL)
        dserror("Dynamic cast to CrosslinkerNode failed");
      if(crosslinker_i->NumElement() != 1)
        dserror("More than one element for this crosslinker");
#endif
    const int clcollid = crosslinker_i->LID();

    CrosslinkerData& cldata_i = crosslinker_data_[clcollid];

    // different treatment according to number of bonds a crosslinker has
    switch(cldata_i.clnumbond)
    {
      case 0:
      {
        // crosslinker has zero bonds, i.e. is free to diffuse according to
        // brownian dynamics
        DiffuseUnboundCrosslinker(crosslinker_i);
        break;
      }
      case 1:
      {
        // get clbspot that is currently bonded
        int occbspotid = 0;
        GetSingleOccupiedClBspot(occbspotid, cldata_i.clbspots);

        // get current position of binding spot of filament partner
        // note: we can not use our beam data container, as bspot position is not current position (as this
        // is the result of a sum, you can not have a reference to that)
        const int elegid = cldata_i.clbspots[occbspotid].first;

#ifdef DEBUG
        // safety check
        const int colelelid = DiscretPtr()->ElementColMap()->LID(elegid);
        if(colelelid<0)
          dserror("Crosslinker has %i bonds but his binding partner with gid %i "
                  "is \nnot ghosted/owned on proc %i (owner of crosslinker)",cldata_i.clnumbond,elegid,GState().GetMyRank());
#endif

        DRT::ELEMENTS::Beam3Base* ele =
            dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->gElement(elegid));

#ifdef DEBUG
        // safety check
        if( ele == NULL)
          dserror("Dynamic cast of ele with gid %i failed on proc ", elegid, GState().GetMyRank());
#endif

        // get current position of filament binding spot
        LINALG::Matrix<3,1> bbspotpos;
        std::vector<double> eledisp;
        BIOPOLYNET::UTILS::GetCurrentElementDis(Discret(),ele,BeamInteractionDataStatePtr()->GetMutableDisColNp(),eledisp);
        ele->GetPosOfBindingSpot(bbspotpos,eledisp,cldata_i.clbspots[occbspotid].second,
            PeriodicBoundingBoxPtr() );

        SetCrosslinkerPosition(crosslinker_i, bbspotpos);

        break;
      }
      case 2:
      {
        // crosslinker has two bonds (cl gets current mid position between the filament
        // binding spot it is attached to)
        // -----------------------------------------------------------------
        // partner one
        // -----------------------------------------------------------------
        int elegid = cldata_i.clbspots[0].first;

#ifdef DEBUG
        if(elegid < 0 or cldata_i.clbspots[0].second < 0)
          dserror(" double bonded crosslinker has stored beam partner gid or loc bsponum of -1, "
                  " something went wrong");
        // safety check
        int colelelid = DiscretPtr()->ElementColMap()->LID(elegid);
        if(colelelid<0)
          dserror("Crosslinker has %i bonds but his binding partner with gid %i "
                  "is not \nghosted/owned on proc %i (owner of crosslinker)",cldata_i.clnumbond,elegid,GState().GetMyRank());
#endif

        DRT::ELEMENTS::Beam3Base* ele =
            dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->gElement(elegid));

#ifdef DEBUG
        // safety check
        if( ele == NULL)
          dserror("Dynamic cast of ele with gid %i failed on proc ", elegid, GState().GetMyRank());
#endif

        // get current position of filament binding spot
        LINALG::Matrix<3,1> bbspotposone;
        std::vector<double> eledisp;
        BIOPOLYNET::UTILS::GetCurrentElementDis(Discret(),ele,BeamInteractionDataStatePtr()->GetMutableDisColNp(),eledisp);
        ele->GetPosOfBindingSpot( bbspotposone, eledisp, cldata_i.clbspots[0].second,
            PeriodicBoundingBoxPtr() );

        // -----------------------------------------------------------------
        // partner two
        // -----------------------------------------------------------------
        elegid = cldata_i.clbspots[1].first;

#ifdef DEBUG
        // safety check
        if(elegid < 0 or cldata_i.clbspots[1].second < 0)
          dserror(" double bonded crosslinker has stored beam partner gid or loc bsponum of -1, "
                  " something went wrong");
        colelelid = DiscretPtr()->ElementColMap()->LID(elegid);
        if(colelelid<0)
          dserror("Crosslinker has %i bonds but his binding partner with gid %i "
                  "is \nnot ghosted/owned on proc %i (owner of crosslinker)",cldata_i.clnumbond,elegid,GState().GetMyRank());
#endif

        ele = dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->gElement(elegid));

#ifdef DEBUG
        // safety check
        if( ele == NULL)
          dserror("Dynamic cast of ele with gid %i failed on proc ", elegid, GState().GetMyRank());
#endif

        // get current position of filament binding spot
        LINALG::Matrix<3,1> bbspotpostwo;
        BIOPOLYNET::UTILS::GetCurrentElementDis( Discret(), ele,
            BeamInteractionDataStatePtr()->GetMutableDisColNp(), eledisp);
        ele->GetPosOfBindingSpot(bbspotpostwo,eledisp,cldata_i.clbspots[1].second,
            PeriodicBoundingBoxPtr() );

        SetPositionOfDoubleBondedCrosslinkerPBCconsistent( crosslinker_i, cldata_i.clpos,
            bbspotposone, bbspotpostwo );

        break;
      }
      default:
      {
        dserror("Unrealistic number %i of bonds for a crosslinker.", cldata_i.clnumbond);
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::DiffuseUnboundCrosslinker(
    DRT::Node* crosslinker) const
{
  CheckInit();

  // diffuse crosslinker according to brownian dynamics
  std::vector<double> randvec;
  int count = 3;
  DRT::Problem::Instance()->Random()->Normal(randvec,count);
  // note: check for compliance with periodic boundary conditions is
  // done during crosslinker transfer in UpdateBinStrategy()
  crosslinker->ChangePos(randvec);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::GetSingleOccupiedClBspot(
    int& occbspotid,
    const std::vector<std::pair<int, int> >& clbspots) const
{
  CheckInit();

  if(clbspots[0].first > -1)
    occbspotid = 0;
  else if(clbspots[1].first > -1)
    occbspotid = 1;
  else
    dserror("numbond = 1 but both binding spots store invalid element GIDs!");

#ifdef DEBUG
  // some safety checks
  if( clbspots[0].first > -1 and clbspots[1].first > -1 )
    dserror("clnumbond should be two not one.");
  if( clbspots[occbspotid].second == -1 )
    dserror(" occupied crosslinker bspot has invalid loc number of a beam binding spot");
  if ( clbspots[0].second > -1 and clbspots[1].second > -1 )
    dserror("major thing wrong with binding status of crosslinker");
#endif

}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::
    SetPositionOfDoubleBondedCrosslinkerPBCconsistent(
        DRT::Node* crosslinker,
        LINALG::Matrix<3,1>& clpos,
        const LINALG::Matrix<3,1>& bspot1pos,
        const LINALG::Matrix<3,1>& bspot2pos ) const
{
  /* the position of (the center) of a double-bonded crosslinker is defined as
   * midpoint between the two given binding spot positions. (imagine a linker
   * being a slender body with a binding domain at each of both ends) */

  /* if the two binding spots are separated by a periodic boundary, we need to
   * shift one position back to get the interpolation right */
  clpos = bspot2pos;
  PeriodicBoundingBox().UnShift3D(clpos,bspot1pos);

  // fixme: to avoid senseless dserror
  LINALG::Matrix<3,1> dummy(clpos);
  clpos.Update(0.5,bspot1pos,0.5,dummy);

  // shift the interpolated position back in the periodic box if necessary
  PeriodicBoundingBox().Shift3D(clpos);

  SetCrosslinkerPosition(crosslinker,clpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::SetCrosslinkerPosition(
    DRT::Node* crosslinker,
    const LINALG::Matrix<3,1>& newclpos) const
{
  CheckInit();

  std::vector<double> newpos(3,0.0);
  for(int dim=0; dim<3; ++dim)
    newpos[dim] = newclpos(dim);
  crosslinker->SetPos(newpos);

}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::SetPositionOfNewlyFreeCrosslinker(
    DRT::Node* crosslinker,
    LINALG::Matrix<3,1>& clpos) const
{
  CheckInit();

  //generate vector in random direction
  // of length half the linking length to "reset" crosslink molecule position: it may now
  // reenter or leave the bonding proximity
  // todo: does this make sense?
  LINALG::Matrix<3,1> cldeltapos_i;
  std::vector<double> randunivec(3);
  int count = 3;
  DRT::Problem::Instance()->Random()->Uni(randunivec, count);
  for (int dim=0; dim<3; ++dim)
    cldeltapos_i(dim) = randunivec[dim];
  cldeltapos_i.Scale(0.5*crosslinking_params_ptr_->LinkingLength() / cldeltapos_i.Norm2());

  clpos.Update(1.0,cldeltapos_i,1.0);

  PeriodicBoundingBox().Shift3D(clpos);
  SetCrosslinkerPosition(crosslinker, clpos);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PreComputeCrosslinkerData()
{
  CheckInit();

  crosslinker_data_.clear();
  const int numcolcl = BinDiscretPtr()->NumMyColNodes();
  crosslinker_data_.resize(numcolcl);

  for ( int i = 0; i < numcolcl; ++i )
  {
    // crosslinker i for which data will be collected
    CROSSLINKING::CrosslinkerNode *crosslinker_i =
        dynamic_cast<CROSSLINKING::CrosslinkerNode*>(BinDiscretPtr()->lColNode(i));

#ifdef DEBUG
      if(crosslinker_i == NULL)
        dserror("Dynamic cast to CrosslinkerNode failed");
      if(crosslinker_i->NumElement() != 1)
        dserror("More than one element for this crosslinker");
#endif

    // store data of crosslinker i according to column lid
    CrosslinkerData& cldata = crosslinker_data_[i];

    // store positions
    for(int dim=0; dim<3; ++dim)
      cldata.clpos(dim) = crosslinker_i->X()[dim];
    // get current binding spot status of crosslinker
    cldata.clbspots = crosslinker_i->ClData()->GetClBSpotStatus();
    // get current binding spot status of crosslinker
    cldata.bnodegids_ = crosslinker_i->ClData()->GetBeamNodeGids();
    // get number of bonds
    cldata.clnumbond = crosslinker_i->ClData()->GetNumberOfBonds();
    // get type of crosslinker (i.e. its material)
    cldata.clmat = crosslinker_i->GetMaterial();
    // get owner
    cldata.clowner = crosslinker_i->Owner();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PreComputeBeamData()
{
  CheckInit();

  beam_data_.clear();
  const int numcolbeams = DiscretPtr()->NumMyColElements();
  beam_data_.resize(numcolbeams);

  // loop over all column beams elements
  for ( int i = 0; i < numcolbeams; ++i )
  {
    // beam element i for which data will be collected
    DRT::ELEMENTS::Beam3Base* beamele_i =
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->lColElement(i));

#ifdef DEBUG
      if(beamele_i == NULL)
        dserror("Dynamic cast to Beam3Base failed");
#endif

    // store data
    BeamData& bdata = beam_data_[i];

    std::vector<double> eledisp;
    BIOPOLYNET::UTILS::GetCurrentElementDis( Discret(), beamele_i,
        BeamInteractionDataStatePtr()->GetMutableDisColNp(), eledisp );

    // loop over all binding spots of current element
    const int numbbspot = static_cast<int>(beamele_i->GetBindingSpotStatus().size());
    for(int j=0; j<numbbspot; ++j)
      BIOPOLYNET::UTILS::GetPosAndTriadOfBindingSpot( beamele_i, BeamInteractionDataStatePtr()->GetMutableDisColNp(),
          PeriodicBoundingBoxPtr(), j, bdata.bbspotpos[j], bdata.bbspottriad[j], eledisp );

    // get status of beam binding spots
    bdata.bbspotstatus = beamele_i->GetBindingSpotStatus();
    // get owner
    bdata.bowner = beamele_i->Owner();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::FindPotentialBindingEvents(
    std::map<int, BindEventData >&              mybonds,
    std::map<int, std::vector<BindEventData> >& undecidedbonds)
{
  CheckInit();

  TEUCHOS_FUNC_TIME_MONITOR("STR::MODELEVALUATOR::Crosslinking::FindPotentialBindingEvents");

  // gather data for all column crosslinker and column beams initially
  PreComputeCrosslinkerData();
  PreComputeBeamData();

  // this variable is used to check if a beam binding spot is linked twice on
  // myrank during a time step
  std::vector< std::set <int> > intendedbeambonds( DiscretPtr()->NumMyRowElements() );

  // store bins that have already been examined
  std::set<int> examinedbins;
  // loop over all column crosslinker in random order
  // create random order of indices
  std::vector<int> rordercolcl =
      BIOPOLYNET::UTILS::Permutation(BinDiscretPtr()->NumMyColNodes());
  std::vector<int>::const_iterator icl;
  for( icl = rordercolcl.begin(); icl != rordercolcl.end(); ++icl )
  {
    DRT::Node *currcrosslinker = BinDiscretPtr()->lColNode( *icl );

#ifdef DEBUG
      if(currcrosslinker == NULL)
        dserror("Dynamic cast to CrosslinkerNode failed");
      if(currcrosslinker->NumElement() != 1)
        dserror("More than one element for this crosslinker");
#endif

    // get bin that contains this crosslinker (can only be one)
    DRT::Element* CurrentBin = currcrosslinker->Elements()[0];
    const int currbinId = CurrentBin->Id();

#ifdef DEBUG
      if(currbinId < 0 )
        dserror(" negative bin id number %i ", currbinId );
#endif

    // if a bin has already been examined --> continue with next crosslinker
    if(examinedbins.find(currbinId) != examinedbins.end())
      continue;
    //else: bin is examined for the first time --> new entry in examinedbins_
    else
      examinedbins.insert(currbinId);

    // get neighboring bins
    // note: interaction distance cl to beam needs to be smaller than half bin size
    std::vector<int> neighboring_binIds;
    neighboring_binIds.reserve(27);
    // do not check on existence here -> shifted to GetBinContent
    ParticleHandlerPtr()->BinStrategy()->GetNeighborAndOwnBinIds(
        currbinId, neighboring_binIds );

    // get set of neighbouring beam elements (i.e. elements that somehow touch nb bins)
    // as explained above, we only need row elements
    std::set<DRT::Element*> neighboring_beams;
    ParticleHandlerPtr()->GetBinContent( neighboring_beams, bin_beamcontent_,
        neighboring_binIds, true );

    // get all crosslinker in current bin
    DRT::Node **ClInCurrentBin = CurrentBin->Nodes();
    const int numcrosslinker = CurrentBin->NumNode();

    // obtain random order in which crosslinker are addressed
    std::vector<int> randorder = BIOPOLYNET::UTILS::Permutation( numcrosslinker );

    // loop over all crosslinker in CurrentBin in random order
    std::vector<int>::const_iterator randcliter;
    for( randcliter = randorder.begin(); randcliter != randorder.end(); ++randcliter )
    {
      // get random crosslinker in current bin
      DRT::Node *crosslinker_i = ClInCurrentBin[*randcliter];
      // get all potential binding events on myrank
      PrepareBinding( crosslinker_i, neighboring_beams, mybonds, undecidedbonds, intendedbeambonds );
    }
  }

}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PrepareBinding(
    DRT::Node*                                  crosslinker_i,
    const std::set<DRT::Element*>&              neighboring_beams,
    std::map<int, BindEventData >&              mybonds,
    std::map<int, std::vector<BindEventData> >& undecidedbonds,
    std::vector< std::set <int> >&              intendedbeambonds)
{
  CheckInit();

  // get precomputed data of crosslinker i
  const int clcollid = crosslinker_i->LID();
  CrosslinkerData const& cldata_i = crosslinker_data_[clcollid];

  // -------------------------------------------------------------------------
  // We now check all criteria that need to be passed for a binding event one
  // after the other
  // -------------------------------------------------------------------------

  // 1. criterion: in case crosslinker is double bonded, we can leave here
  if( cldata_i.clnumbond == 2 )
    return;

  // if crosslinker is singly bound, we fetch the orientation vector
  LINALG::Matrix<3,1> occ_bindingspot_beam_tangent(true);
  if( cldata_i.clnumbond == 1 )
    GetOccupiedClBSpotBeamTangent( cldata_i, occ_bindingspot_beam_tangent );

  // minimum and maximum distance at which a double-bond crosslink can be established
  // todo: this needs to go crosslinker material
  double linkdistmin = crosslinking_params_ptr_->LinkingLength()
                      - crosslinking_params_ptr_->LinkingLengthTolerance();
  double linkdistmax = crosslinking_params_ptr_->LinkingLength()
                      + crosslinking_params_ptr_->LinkingLengthTolerance();

  double linkanglemin = crosslinking_params_ptr_->LinkingAngle()
                      - crosslinking_params_ptr_->LinkingAngleTolerance();
  double linkanglemax = crosslinking_params_ptr_->LinkingAngle()
                      + crosslinking_params_ptr_->LinkingAngleTolerance();

  // probability with which a crosslinker is established between crosslink
  // molecule and neighbor binding spot
  const double kon = crosslinking_params_ptr_->KOn();
  double plink = 1.0 - exp( -(*GState().GetDeltaTime())[0]* kon);

  // -------------------------------------------------------------------------
  // look for potential interaction of crosslinker i and a binding spot of an
  // element, i.e. distance \Delta = R +/- \Delta R
  // -------------------------------------------------------------------------
  // loop over all neighboring beam elements in random order (keep in mind
  // we are only looping over row elements)
  std::vector<DRT::Element*> beamvec( neighboring_beams.begin(), neighboring_beams.end() );
  const int numbeams = beamvec.size();
  std::vector<int> randorder = BIOPOLYNET::UTILS::Permutation(numbeams);
  std::vector<int> ::const_iterator randiter;
  for(randiter=randorder.begin(); randiter!=randorder.end();  ++randiter)
  {
    // get neighboring (nb) beam element
    DRT::ELEMENTS::Beam3Base* nbbeam =
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(beamvec[*randiter]);

#ifdef DEBUG
      if(nbbeam == NULL)
        dserror("Dynamic cast to beam3base failed");
#endif

    // get pre computed data of current nbbeam
    BeamData const& beamdata_i = beam_data_[ nbbeam->LID() ];

#ifdef DEBUG
      if( nbbeam->LID() < 0 )
        dserror("Beami lid < 0");
#endif

    if( cldata_i.clnumbond == 1 )
    {
      int occbspotid = 0;
      GetSingleOccupiedClBspot(occbspotid, cldata_i.bnodegids_);

      // 2. criterion:
      // exclude binding of a single bonded crosslinker in close proximity on the
      // same filament (i.e. element cloud of old element binding partner is excluded)
      if(CheckCrosslinkOfAdjacentElements( nbbeam, cldata_i.bnodegids_[occbspotid] ) )
        // got to next neighboring element
        continue;
    }

    // loop over all binding spots of current element in random order
    std::vector<int> randbspot = BIOPOLYNET::UTILS::Permutation(beamdata_i.bbspotstatus.size());
    std::vector<int> ::const_iterator rbspotiter;
    for( rbspotiter = randbspot.begin(); rbspotiter != randbspot.end(); ++rbspotiter )
    {
      // get local number of binding spot in element
      const int locnbspot = *rbspotiter;

      // get current position and tangent vector of filament axis at free binding spot
      LINALG::Matrix<3,1> const& currbbspos = beamdata_i.bbspotpos.at(locnbspot);

      // note: we use first base vector instead of tangent vector here
      LINALG::Matrix<3,1> curr_bindingspot_beam_tangent(true);
      for ( unsigned int idim = 0; idim < 3; ++idim )
        curr_bindingspot_beam_tangent(idim) = beamdata_i.bbspottriad.at(locnbspot)(idim,0);

      // -----------------------------------------------------------------------
      // we are now doing some additional checks if a binding event can happen
      // -----------------------------------------------------------------------
      {
        // 3. criterion:
        // first check if binding spot is free, if not, check next bspot on curr ele
        // note: bspotstatus in bonded case holds cl gid, otherwise -1 (meaning free)
        if(beamdata_i.bbspotstatus.at(locnbspot) != -1)
          continue;

        // 4. criterion: check RELEVANT distance criterion
        // if free:
        // distance between crosslinker center and current beam binding spot
        // if singly bound:
        // distance between already bound bspot of crosslinker and current beam binding spot
        // note: as we set the crosslinker position to coincide with beam bspot position if singly bound,
        //       we can also use cldata_i.clpos in the second case

        if( (cldata_i.clnumbond == 0 and
             IsDistanceOutOfBindingRange(cldata_i.clpos,currbbspos,0.5*linkdistmin,0.5*linkdistmax)) or
            (cldata_i.clnumbond == 1 and
             IsDistanceOutOfBindingRange(cldata_i.clpos,currbbspos,linkdistmin,linkdistmax)) )
          continue;

        // 5. criterion: orientation of centerline tangent vectors at binding spots
        // a crosslink (double-bonded crosslinker) will only be established if the
        // enclosed angle is in the specified range
        if( cldata_i.clnumbond == 1 and
           IsEnclosedAngleOfBSpotTangentsOutOfRange( occ_bindingspot_beam_tangent,
               curr_bindingspot_beam_tangent, linkanglemin, linkanglemax ) )
          continue;

        // 6. criterion:
        // a crosslink is set if and only if it passes the probability check
        // for a binding event to happen
        if( DRT::Problem::Instance()->Random()->Uni() > plink )
          continue;

        // 7. criterion
        // check if current beam binding spot yet intended to bind this timestep
        // by a crosslinker that came before in this random order
        int const beamrowlid = DiscretPtr()->ElementRowMap()->LID( nbbeam->Id() );
        if ( intendedbeambonds[beamrowlid].find( locnbspot ) != intendedbeambonds[beamrowlid].end() )
        {
          /* note: it is possible that the binding event that rejects the current one is rejected itself
           * later during communication with other procs and therefore the current one could be
           * valid. Just neclecting this here is a slight inconsistency, but should be ok as such an
           * coincidence is extremely rare in a simulation with realistic proportion of crosslinker
           * to beam binding spots. Additionally missing one event would not change any physics.
           * ( Could be cured with additional communication)
           */
          if ( DiscretPtr()->Comm().NumProc() > 1 and GState().GetMyRank() == 0 )
            std::cout << " Warning: There is a minimal chance of missing a regular binding event" << std::endl;
          continue;
        }
        else
        {
          // insert current event
          intendedbeambonds[beamrowlid].insert(locnbspot);
        }
      }


      // ---------------------------------------------------------------------
      // if we came this far, we can add this potential binding event to its
      // corresponding map
      // ---------------------------------------------------------------------
      BindEventData bindeventdata;
      bindeventdata.clgid = crosslinker_i->Id();
      bindeventdata.elegid = nbbeam->Id();
      bindeventdata.bspotlocn = locnbspot;
      bindeventdata.requestproc = GState().GetMyRank();
      // this is default, is changed if owner of cl has something against it
      bindeventdata.permission = 1;

      // in case myrank is owner, we add it to the mybonds map
      if( cldata_i.clowner == GState().GetMyRank() )
      {
        mybonds[bindeventdata.clgid] = bindeventdata;
      }
      // myrank is not owner, we add it to the map of events that need to be
      // communicated to make a decision
      else
      {
        undecidedbonds[cldata_i.clowner].push_back(bindeventdata);
      }

      // as we allow only one binding event for each cl in one time step,
      // we are done here, if we made it so far (i.e met 1., 2., 3., 4. and 5. criterion)
      return;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::GetOccupiedClBSpotBeamTangent(
    CrosslinkerData const& cldata_i,
    LINALG::Matrix<3,1>& occ_bindingspot_beam_tangent) const
{
  CheckInitSetup();

  int occbspotid = 0;
  GetSingleOccupiedClBspot(occbspotid, cldata_i.bnodegids_);

  const int elegid = cldata_i.clbspots[occbspotid].first;
  const int elecollid = Discret().ElementColMap()->LID(elegid);
  const int locbspotnum = cldata_i.clbspots[occbspotid].second;

#ifdef DEBUG
  if( elecollid < 0 )
    dserror (" Element with gid %i you are looking for on rank %i not even ghosted",
             elegid, GState().GetMyRank() );
#endif

  // note: we use first base vector instead of tangent vector here
  for ( unsigned int idim = 0; idim < 3; ++idim )
    occ_bindingspot_beam_tangent(idim) =
        beam_data_[elecollid].bbspottriad.at(locbspotnum)(idim,0);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::CheckCrosslinkOfAdjacentElements(
    DRT::Element* nbbeam,
    const std::pair<int, int>& occbspot_bnodegids) const
{
  CheckInit();

  // check if two considered eles share nodes
  for ( int i = 0; i < 2; ++i )
    if( nbbeam->NodeIds()[i] == occbspot_bnodegids.first ||
        nbbeam->NodeIds()[i] == occbspot_bnodegids.second)
      return true;

  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::IsDistanceOutOfBindingRange(
    const LINALG::Matrix<3,1>& pos1,
    const LINALG::Matrix<3,1>& pos2,
    const double& lowerbound,
    const double& upperbound
    ) const
{
  LINALG::Matrix<3,1> dist_vec(true);
  dist_vec.Update(1.0, pos1, -1.0, pos2);

  const double distance = dist_vec.Norm2();

  if (distance < lowerbound or distance > upperbound)
    return true;
  else
    return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::IsEnclosedAngleOfBSpotTangentsOutOfRange(
    const LINALG::Matrix<3,1>& direction1,
    const LINALG::Matrix<3,1>& direction2,
    const double& lowerbound,
    const double& upperbound
    ) const
{
  // cosine of angle is scalar product of vectors divided by their norms
  // direction vectors should be unit vectors since they come from triads, but anyway ...
  double cos_angle = direction1.Dot(direction2) / direction1.Norm2() / direction2.Norm2();

  if (cos_angle>1.0)
    dserror("cos(angle) = %f > 1.0 ! restrict this to exact 1.0 to avoid NaN in "
        "following call to std::acos",cos_angle);

  double angle = std::acos(cos_angle);

  // acos returns angle \in [0,\pi] but we always want the acute angle here
  if (angle > 0.5*M_PI)
    angle = M_PI - angle;

  if (angle < lowerbound or angle > upperbound)
    return true;
  else
    return false;
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ManageBindingInParallel(
    std::map<int, BindEventData >&              mybonds,
    std::map<int, std::vector<BindEventData> >& undecidedbonds,
    std::map<int, BindEventData >&              myelebonds) const
{
  CheckInit();

  TEUCHOS_FUNC_TIME_MONITOR("STR::MODELEVALUATOR::Crosslinking::ManageBindingInParallel");

  // variable for safety check
  int numrecrequest;
  // exporter
  DRT::Exporter exporter( BinDiscret().Comm() );

  // -------------------------------------------------------------------------
  // 1) each procs makes his requests and receives the request of other procs
  // -------------------------------------------------------------------------
  // store requested cl and its data
  std::map<int, std::vector<BindEventData> > requestedcl;
  CommunicateUndecidedBonds( exporter, undecidedbonds, numrecrequest, requestedcl );

  // -------------------------------------------------------------------------
  // 2) now myrank needs to decide which proc is allowed to set the requested
  //    link
  // -------------------------------------------------------------------------
  std::map<int, std::vector<BindEventData> > decidedbonds;
  DecideBindingInParallel( requestedcl, mybonds, decidedbonds );

  // -------------------------------------------------------------------------
  // 3) communicate the binding decisions made on myrank, receive decisions
  //    made for its own requests and create colbondmap accordingly
  // -------------------------------------------------------------------------
  int answersize = static_cast<int>(undecidedbonds.size());
  CommunicateDecidedBonds( exporter, decidedbonds, myelebonds, numrecrequest, answersize );
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::BindMyCrosslinkerAndElements(
    std::map<int, BindEventData >& mybonds,
    std::map<int, BindEventData >& myelebonds)
{
  CheckInit();

  TEUCHOS_FUNC_TIME_MONITOR("STR::MODELEVALUATOR::Crosslinking::BindCrosslinker");

  // map key is crosslinker gid to be able to uniquely address one entry over all procs
  std::map<int, NewDoubleBonds> mynewdbondcl;

  // myrank owner of crosslinker and most elements
  BindMyCrosslinker( mybonds, mynewdbondcl );

  // myrank only owner of current binding partner ele
  BindMyElements( myelebonds );

  // setup new double bonds and insert them in doublebondcl_
  SetupNewDoubleBonds( mynewdbondcl );
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::BindMyCrosslinker(
    std::map<int, BindEventData > const& mybonds,
    std::map<int, NewDoubleBonds>&       mynewdbondcl)
{
  CheckInit();

  std::map<int, BindEventData>::const_iterator cliter;
  for( cliter = mybonds.begin(); cliter != mybonds.end(); ++cliter )
  {
    // get binding event data
    BindEventData binevdata = cliter->second;

#ifdef DEBUG
    if( binevdata.permission != 1)
      dserror(" Rank %i wants to bind crosslinker %i without permission, "
              " something went wrong", GState().GetMyRank(), cliter->first);
#endif

    // get current linker
    const int clcollid = BinDiscretPtr()->NodeColMap()->LID(cliter->first);

#ifdef DEBUG
    // safety checks
    if(clcollid<0)
      dserror("Crosslinker not even ghosted, should be owned here.");
#endif

    CROSSLINKING::CrosslinkerNode *crosslinker_i =
        dynamic_cast<CROSSLINKING::CrosslinkerNode*>(BinDiscretPtr()->lColNode(clcollid));
    // get crosslinker data
    CrosslinkerData& cldata_i = crosslinker_data_[clcollid];

    // get beam data
    const int colelelid = DiscretPtr()->ElementColMap()->LID(binevdata.elegid);
#ifdef DEBUG
    // safety checks
    if(colelelid<0)
      dserror("Element with gid %i not ghosted.", binevdata.elegid );
#endif

    DRT::ELEMENTS::Beam3Base* beamele_i =
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->lColElement(colelelid));

#ifdef DEBUG
    // safety checks
    if( beamele_i == NULL )
      dserror(" Dynamic cast to beam3base failed " );
#endif

    BeamData& beamdata_i = beam_data_[colelelid];

#ifdef DEBUG
    // safety checks
    if(cliter->first != binevdata.clgid)
      dserror("Map key does not match crosslinker gid of current binding event.");

    if( cldata_i.clowner != GState().GetMyRank() )
      dserror("Only row owner of crosslinker is changing its status");

    if( colelelid < 0 )
      dserror("Binding element partner of current row crosslinker is not ghosted, "
              "this must be the case though.");
#endif

    // -------------------------------------------------------------------------
    // different treatment according to number of bonds crosslinker had before
    // this binding event
    // -------------------------------------------------------------------------
    switch( cldata_i.clnumbond)
    {
      case 0:
      {
        // -----------------------------------------------------------------
        // update crosslinker status
        // -----------------------------------------------------------------
        // store gid and bspot local number of this element, first binding spot
        // always bonded first
        cldata_i.clbspots[0].first  = binevdata.elegid;
        cldata_i.clbspots[0].second = binevdata.bspotlocn;
        crosslinker_i->ClData()->SetClBSpotStatus( cldata_i.clbspots );

        // store gid of first and second node of new binding partner
        cldata_i.bnodegids_[0].first  = beamele_i->NodeIds()[0];
        cldata_i.bnodegids_[0].second = beamele_i->NodeIds()[1];
        crosslinker_i->ClData()->SetBeamNodeGids( cldata_i.bnodegids_ );

        // update number of bonds
        cldata_i.clnumbond = 1;
        crosslinker_i->ClData()->SetNumberOfBonds( cldata_i.clnumbond );

        // update position
        cldata_i.clpos = beamdata_i.bbspotpos[ binevdata.bspotlocn ];
        SetCrosslinkerPosition( crosslinker_i, cldata_i.clpos );

        // -----------------------------------------------------------------
        // update beam status
        // -----------------------------------------------------------------
        // store crosslinker gid in status of beam binding spot if myrank
        // is owner of beam
        if( beamdata_i.bowner == GState().GetMyRank() )
        {
          beamdata_i.bbspotstatus[ binevdata.bspotlocn ] = binevdata.clgid;
          beamele_i->SetBindingSpotStatus( beamdata_i.bbspotstatus );
        }

#ifdef DEBUG
        // safety check
        if(not (cldata_i.clbspots[1].first < 0) )
          dserror("Numbond does not fit to clbspot vector.");
#endif

        break;
      }
      case 1:
      {
        // get clbspot that is currently bonded
        int occbspotid = 0;
        GetSingleOccupiedClBspot(occbspotid, cldata_i.clbspots);
        int freebspotid = 1;
        if( occbspotid == 1 )
          freebspotid = 0;

        // -----------------------------------------------------------------
        // update crosslinker status
        // -----------------------------------------------------------------
        // store gid and bspot local number of this element
        cldata_i.clbspots[freebspotid].first = binevdata.elegid;
        cldata_i.clbspots[freebspotid].second = binevdata.bspotlocn;
        crosslinker_i->ClData()->SetClBSpotStatus(cldata_i.clbspots);

        // store gid of first and second node of this element
        cldata_i.bnodegids_[freebspotid].first = beamele_i->NodeIds()[0];
        cldata_i.bnodegids_[freebspotid].second = beamele_i->NodeIds()[1];
        crosslinker_i->ClData()->SetBeamNodeGids(cldata_i.bnodegids_);

        // update number of bonds
        cldata_i.clnumbond = 2;
        crosslinker_i->ClData()->SetNumberOfBonds(cldata_i.clnumbond);

        // update position
        const LINALG::Matrix<3,1> occbspotpos_copy = cldata_i.clpos;
        SetPositionOfDoubleBondedCrosslinkerPBCconsistent(
            crosslinker_i,
            cldata_i.clpos,
            beamdata_i.bbspotpos[ cldata_i.clbspots[freebspotid].second ],
            occbspotpos_copy);

        // create double bond cl data
        NewDoubleBonds dbondcl;
        dbondcl.id = binevdata.clgid;
        if( cldata_i.clbspots[freebspotid].first > cldata_i.clbspots[occbspotid].first)
        {
          dbondcl.eleids.push_back( cldata_i.clbspots[freebspotid] );
          dbondcl.eleids.push_back( cldata_i.clbspots[occbspotid] );
        }
        else
        {
          dbondcl.eleids.push_back( cldata_i.clbspots[occbspotid] );
          dbondcl.eleids.push_back( cldata_i.clbspots[freebspotid] );
        }

        // insert pair in mypairs
        mynewdbondcl[dbondcl.id] = dbondcl;

        // first check if myrank is owner of element of current binding event
        // (additionally to being owner of cl)
        if(beamdata_i.bowner == GState().GetMyRank())
        {
          // update beam data
          // store crosslinker gid in status of beam binding spot
          beamdata_i.bbspotstatus[ binevdata.bspotlocn ] = binevdata.clgid;
          beamele_i->SetBindingSpotStatus(beamdata_i.bbspotstatus);
        }

        break;
      }
      default:
      {
        dserror("You should not be here, crosslinker has unrealistic number "
                "%i of bonds.", cldata_i.clnumbond);
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::BindMyElements(
    const std::map<int, BindEventData >& myelebonds)
{
  CheckInit();

  /*
   * 1| 2__|2  or  1| 3__|2  or  1| 2__|1
   * 1|    |2      1|    |2      1|    |1
   * legend: | = beam; __= cl; 2,3 = owner; 1=myrank
   */
  // loop through all binding events
  std::map<int, BindEventData>::const_iterator cliter;
  for( cliter = myelebonds.begin(); cliter != myelebonds.end(); ++cliter )
  {
    // get binding event data
    BindEventData binevdata = cliter->second;

    // get linker data
    const int clcollid = BinDiscretPtr()->NodeColMap()->LID(cliter->first);
#ifdef DEBUG
    // safety check
    if( clcollid < 0 )
     dserror("Crosslinker needs to be ghosted, but this isn't the case.");
#endif
    CrosslinkerData& cldata_i = crosslinker_data_[clcollid];

    // get beam data
    const int colelelid = DiscretPtr()->ElementColMap()->LID(binevdata.elegid);
#ifdef DEBUG
    // safety check
    if(colelelid<0)
     dserror("element with gid %i not ghosted on proc %i",binevdata.elegid,GState().GetMyRank());
#endif
    // get beam element of current binding event
    DRT::ELEMENTS::Beam3Base* ele_i =
       dynamic_cast<DRT::ELEMENTS::Beam3Base*>(DiscretPtr()->lColElement(colelelid));
    BeamData& beamdata_i = beam_data_[colelelid];

#ifdef DEBUG
    // safety checks
    if( beamdata_i.bowner != GState().GetMyRank() )
     dserror("Only row owner of element is allowed to change its status");
    if( cldata_i.clowner == GState().GetMyRank() )
     dserror("myrank should not be owner of this crosslinker");
#endif

    // different treatment according to number of bonds crosslinker had before
    // this binding event
    switch( cldata_i.clnumbond )
    {
      case 0:
      {
        // update beam data
        // store crosslinker gid in status of beam binding spot
        beamdata_i.bbspotstatus[ binevdata.bspotlocn ] = binevdata.clgid;
        ele_i->SetBindingSpotStatus( beamdata_i.bbspotstatus );
        break;
      }
      case 1:
      {
        // update beam data
        // store crosslinker gid in status of beam binding spot
        beamdata_i.bbspotstatus[ binevdata.bspotlocn ] = binevdata.clgid;
        ele_i->SetBindingSpotStatus(beamdata_i.bbspotstatus);

        break;
      }
      default:
      {
        dserror("You should not be here, crosslinker has unrealistic number "
                "%i of bonds.", cldata_i.clnumbond);
        break;
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::SetupNewDoubleBonds(
    std::map<int, NewDoubleBonds> const& mynewdbondcl)
{
  CheckInit();

  std::map<int, NewDoubleBonds>::const_iterator iter;
  for ( iter = mynewdbondcl.begin(); iter != mynewdbondcl.end(); ++iter )
  {
    // init positions and triads
    std::vector<LINALG::Matrix<3,1> > pos(2);
    std::vector<LINALG::Matrix<3,3> > triad(2);

    NewDoubleBonds const& newdoublebond_i = iter->second;

    for( int i = 0; i < 2; ++i )
    {
      int elegid = newdoublebond_i.eleids[i].first;
      int locbspotnum = newdoublebond_i.eleids[i].second;
      DRT::Element* ele = DiscretPtr()->gElement(elegid);

#ifdef DEBUG
      // safety checks
      if( ele == NULL )
        dserror("Element with gid %i not there on rank %i", elegid, GState().GetMyRank() );
#endif

      pos[i] = beam_data_[ele->LID()].bbspotpos.at(locbspotnum);
      triad[i] = beam_data_[ele->LID()].bbspottriad.at(locbspotnum);
    }

    // ToDo specify and pass material parameters for crosslinker element

    // create and initialize objects of beam-to-beam connections
    // Todo introduce enum for type of linkage (only linear Beam3r element possible so far)
    //      and introduce corresponding input parameter or even condition for mechanical
    //      links between beams in general
    Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> linkelepairptr =
      BEAMINTERACTION::BeamToBeamLinkage::Create();

    // finally initialize and setup object
    linkelepairptr->Init( iter->first, newdoublebond_i.eleids, pos, triad );
    linkelepairptr->Setup();

    // add to my double bonds
    doublebondcl_[linkelepairptr->Id()] = linkelepairptr;
  }

  // print some information
  if( mynewdbondcl.size() > 0 or doublebondcl_.size() > 0 )
  {
    IO::cout(IO::standard) <<"\n************************************************"<<IO::endl;
    IO::cout(IO::standard) << "PID " << GState().GetMyRank() << ": added " << mynewdbondcl.size()
        << " new db crosslinkers. Now have " << doublebondcl_.size() <<IO::endl;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UnBindCrosslinker()
{
  CheckInit();

  // todo: this needs to go somewhere else
  //------------------------------------------------------------------------------
  // get current off-rate for crosslinkers
  double koff = crosslinking_params_ptr_->KOff();
  const double dt = (*GState().GetDeltaTime())[0];

  // probability with which a crosslink breaks up in the current time step
  double p_unlink = 1.0 - exp(-dt * koff);

  //------------------------------------------------------------------------------

  // data containing information about elements that need to be updated on
  // procs != myrank
  std::map<int, std::vector<UnBindEventData> > sendunbindevents;
  // elements that need to be updated on myrank
  std::vector<UnBindEventData> myrankunbindevents;

  // loop over all row linker (in random order) and dissolve bond if probability
  // criterion is met
  /* note: we loop over all row crosslinker, i.e. myrank needs to update all
   * crosslinker information. As it possible that a row crosslinker is linked
   * to col element, we potentially need to communicate if such an element
   * needs to be updated*/
  const int numrowcl = BinDiscretPtr()->NumMyRowNodes();
  std::vector<int> rorderrowcl = BIOPOLYNET::UTILS::Permutation(numrowcl);
  std::vector<int>::const_iterator rowcli;
  for( rowcli = rorderrowcl.begin(); rowcli != rorderrowcl.end(); ++rowcli )
  {
    // get current linker
    CROSSLINKING::CrosslinkerNode *crosslinker_i =
        dynamic_cast<CROSSLINKING::CrosslinkerNode*>(BinDiscretPtr()->lRowNode(*rowcli));

    // get linker data
    const int clcollid = crosslinker_i->LID();
    CrosslinkerData& cldata_i = crosslinker_data_[clcollid];

    // different treatment according to number of bonds of a crosslinker
    switch(cldata_i.clnumbond)
    {
      case 0:
      {
        // nothing to do here
        break;
      }
      case 1:
      {
        // if probability criterion is not met, we are done here
        if (DRT::Problem::Instance()->Random()->Uni() > p_unlink)
          break;

        // dissolve bond
        int occbspotid = 0;
        GetSingleOccupiedClBspot(occbspotid, cldata_i.clbspots);

        // -----------------------------------------------------------------
        // update beam status (first check which rank is responsible)
        // -----------------------------------------------------------------

        // store unbinding event data
        UnBindEventData unbindevent;
        unbindevent.eletoupdate = cldata_i.clbspots[occbspotid];

        // owner of beam
        const int beamowner =
            DiscretPtr()->gElement(unbindevent.eletoupdate.first)->Owner();

        // check who needs to update the element status
        if( beamowner == GState().GetMyRank() )
          myrankunbindevents.push_back(unbindevent);
        else
          sendunbindevents[beamowner].push_back(unbindevent);

        // -----------------------------------------------------------------
        // update crosslinker status
        // -----------------------------------------------------------------
        // update binding status of linker
        cldata_i.clbspots[occbspotid].first = -1;
        cldata_i.clbspots[occbspotid].second = -1;
        crosslinker_i->ClData()->SetClBSpotStatus(cldata_i.clbspots);

        // store gid of first and second node of new binding partner
        cldata_i.bnodegids_[occbspotid].first = -1;
        cldata_i.bnodegids_[occbspotid].second = -1;
        crosslinker_i->ClData()->SetBeamNodeGids(cldata_i.bnodegids_);

        // update number of bonds
        cldata_i.clnumbond = 0;
        crosslinker_i->ClData()->SetNumberOfBonds(cldata_i.clnumbond);

        // update position of crosslinker
        SetPositionOfNewlyFreeCrosslinker(crosslinker_i,cldata_i.clpos);

        break;
      }
      case 2:
      {
        // get id of freed and still occupied bspot
        int freedbspotid = -1;
        int stayocc = -1;
        // loop through crosslinker bonds in random order
        std::vector<int> ro = BIOPOLYNET::UTILS::Permutation(cldata_i.clnumbond);
        std::vector<int>::const_iterator clbspotiter;
        for( clbspotiter = ro.begin(); clbspotiter != ro.end(); ++clbspotiter )
        {
          // if probability criterion isn't met, go to next spot
          if (DRT::Problem::Instance()->Random()->Uni() > p_unlink)
            continue;

          // get id of freed and still occupied bspot
          freedbspotid = *clbspotiter;
          stayocc = 0;
          if(freedbspotid == 0)
            stayocc = 1;

#ifdef DEBUG
          // safety check
          if(not doublebondcl_.count(crosslinker_i->Id() ) )
          dserror("willing to delete not existing entry, something went wrong");
#endif

          // erase double bond
          doublebondcl_.erase( crosslinker_i->Id() );

          // -----------------------------------------------------------------
          // update beam status (first check which rank is responsible)
          // -----------------------------------------------------------------
          // initialize ubindevent
          UnBindEventData unbindevent;
          unbindevent.eletoupdate = cldata_i.clbspots[freedbspotid];

          // owner of beam element bond that gets dissolved
          const int freedbeamowner =
              DiscretPtr()->gElement(cldata_i.clbspots[freedbspotid].first)->Owner();

          if( freedbeamowner == GState().GetMyRank() )
            myrankunbindevents.push_back( unbindevent );
          else
            sendunbindevents[freedbeamowner].push_back( unbindevent );

          // -----------------------------------------------------------------
          // update crosslinker status
          // -----------------------------------------------------------------
          // reset binding status of freed crosslinker binding spot
          cldata_i.clbspots[freedbspotid].first = -1;
          cldata_i.clbspots[freedbspotid].second = -1;
          crosslinker_i->ClData()->SetClBSpotStatus(cldata_i.clbspots);

          // store gid of first and second node of new binding partner
          cldata_i.bnodegids_[freedbspotid].first = -1;
          cldata_i.bnodegids_[freedbspotid].second = -1;
          crosslinker_i->ClData()->SetBeamNodeGids(cldata_i.bnodegids_);

          // update number of bonds
          cldata_i.clnumbond = 1;
          crosslinker_i->ClData()->SetNumberOfBonds(cldata_i.clnumbond);

          // update postion
          const int collidoccbeam =
              DiscretPtr()->ElementColMap()->LID(cldata_i.clbspots[stayocc].first);
#ifdef DEBUG
          // safety check
          if( collidoccbeam < 0 )
            dserror("element with gid %i not ghosted on proc %i",cldata_i.clbspots[stayocc].first, GState().GetMyRank() );
#endif
          BeamData& beamdata_i = beam_data_[collidoccbeam];
          cldata_i.clpos = beamdata_i.bbspotpos[cldata_i.clbspots[stayocc].second];
          SetCrosslinkerPosition(crosslinker_i, cldata_i.clpos);

          // we only want to dissolve one bond per timestep, therefore we go to
          // next crosslinker if we made it so far (i.e. a bond got dissolved)
          break;
        }

        break;
      }
      default:
      {
        dserror("Unrealistic number %i of bonds for a crosslinker.", cldata_i.clnumbond);
        break;
      }
    }
  }

  // communicate which elements that need to be updated on rank!=myrank
  CommunicateCrosslinkerUnbinding( sendunbindevents, myrankunbindevents );

  // update binding status of beam binding partners on myrank
  UpdateBeamBindingStatusAfterUnbinding( myrankunbindevents );
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UpdateBeamBindingStatusAfterUnbinding(
    std::vector<UnBindEventData> const& unbindevent )
{
  CheckInit();

  // loop through all unbinding events on myrank
  std::vector<UnBindEventData>::const_iterator iter;
  for(iter=unbindevent.begin(); iter!=unbindevent.end(); ++ iter)
  {
    // get data
    const int elegidtoupdate = iter->eletoupdate.first;
    const int bspotlocn = iter->eletoupdate.second;
    const int colelelid = Discret().ElementColMap()->LID(elegidtoupdate);

#ifdef DEBUG
    // safety check
    if( colelelid < 0 )
      dserror("element with gid %i not owned by proc %i",elegidtoupdate,GState().GetMyRank());
#endif

    // get beam element of current binding event
    DRT::ELEMENTS::Beam3Base* ele_i =
        dynamic_cast<DRT::ELEMENTS::Beam3Base*>(Discret().lColElement(colelelid));

    BeamData& beamdata_i = beam_data_[colelelid];
    std::map<int, int>& bbspotstatus_i = beamdata_i.bbspotstatus;

    // update beam data
    bbspotstatus_i[bspotlocn] = -1;
    ele_i->SetBindingSpotStatus(bbspotstatus_i);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UpdateMyDoubleBondsAfterRedistribution()
{
  CheckInit();

  std::map<int, std::vector<Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> > > dbcltosend;
  std::set<int> dbtoerase;

  // loop over all double bonds on myrank
  std::map<int, Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::iterator iter;;
  for( iter = doublebondcl_.begin(); iter != doublebondcl_.end();)
  {
    const int clgid = iter->first;
    DRT::Node* doublebondedcl_i = BinDiscretPtr()->gNode(clgid);

#ifdef DEBUG
    // safety check
    if(doublebondedcl_i==NULL)
      dserror("Crosslinker moved further than the bin length in one time step, "
              "this is not allowed (maybe increase cutoff radius). ");
#endif

    // check ownership
    int owner = doublebondedcl_i->Owner();
    if(owner != BeamInteractionDataStatePtr()->GetMyRank())
    {
#ifdef DEBUG
      if(not doublebondcl_.count(clgid))
        dserror("willing to delete not existing entry, something went wrong");
#endif
      dbcltosend[owner].push_back(iter->second);
      doublebondcl_.erase(iter++);
    }
    else
    {
      ++iter;
    }
  }

  // add new double bonds
  CommunicateBeamToBeamLinkageAfterRedistribution(dbcltosend);

}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::CommunicateUndecidedBonds(
    DRT::Exporter& exporter,
    std::map<int, std::vector<BindEventData> >& undecidedbonds,
    int& numrecrequest,
    std::map<int, std::vector<BindEventData> >& requestedcl) const
{
  CheckInit();

  // -----------------------------------------------------------------------
  // unblocking send
  // -----------------------------------------------------------------------
  std::vector<MPI_Request> request;
  ISend( exporter, request, undecidedbonds );

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  std::vector<int> summedtargets;
  PrepareReceivingProcs(undecidedbonds,summedtargets);

  numrecrequest = summedtargets[GState().GetMyRank()];
  for(int rec=0; rec<numrecrequest; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.ReceiveAny( from, tag, rdata, length);
    if (tag != 1234)
     dserror("Received on proc %i data with wrong tag from proc %i", GState().GetMyRank(), from);

    // store received data
    std::vector<char>::size_type position = 0;
    while (position < rdata.size())
    {
      // ---- extract received data -----
      BindEventData reccldata;
      UnPack(position,rdata,reccldata);

      // create map holding all requests
      requestedcl[reccldata.clgid].push_back(reccldata);
    }

    if (position != rdata.size())
      dserror("Mismatch in size of data %d <-> %d",static_cast<int>(rdata.size()),position);
  }

  // wait for all communication to finish
  Wait(exporter,request,static_cast<int>(undecidedbonds.size()));

}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::CommunicateDecidedBonds(
    DRT::Exporter& exporter,
    std::map<int, std::vector<BindEventData> >& decidedbonds,
    std::map<int, BindEventData >&              myelebonds,
    const int& numrecrequest,
    const int& answersize) const
{
  CheckInit();

  // -----------------------------------------------------------------------
  // send back decisions for all requests that were made
  // -----------------------------------------------------------------------
  // store requested cl and its data
  std::vector<MPI_Request> request;

#ifdef DEBUG
  // safety check
  if(static_cast<int>(decidedbonds.size()) != numrecrequest)
    dserror("Number of received requests %i unequal to number of answers %i",
        numrecrequest,decidedbonds.size());
#endif

  // unblocking send
  ISend(exporter,request,decidedbonds);

#ifdef DEBUG
  std::vector<int> summedtargets;
  PrepareReceivingProcs(decidedbonds,summedtargets);
  if( answersize != summedtargets[GState().GetMyRank()] )
    dserror(" proc %i did not get an answer to all its questions, that it not fair.");
#endif

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  // store requested cl and its data
  for(int rec=0; rec<answersize; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.ReceiveAny(from,tag,rdata,length);
    if (tag != 1234)
      dserror("Received on proc %i data with wrong tag from proc %i", GState().GetMyRank(), from);

    // store received data
    std::vector<char>::size_type position = 0;
    while (position < rdata.size())
    {
      // ---- extract received data -----
      BindEventData reccldata;
      UnPack(position,rdata,reccldata);

      // add binding events to new colbond map
      if(reccldata.permission)
        myelebonds[reccldata.clgid] = reccldata;

    }

    if (position != rdata.size())
      dserror("Mismatch in size of data %d <-> %d",static_cast<int>(rdata.size()),position);
  }

  // wait for all communication to finish
  Wait(exporter,request,static_cast<int>(decidedbonds.size()));

}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::DecideBindingInParallel(
    std::map<int, std::vector<BindEventData> >& requestedcl,
    std::map<int, BindEventData >&              mybonds,
    std::map<int, std::vector<BindEventData> >& decidedbonds) const
{
  CheckInit();

  std::map<int, std::vector<BindEventData> >::iterator cliter;
  // loop over all requested cl (note myrank is owner of these)
  for( cliter = requestedcl.begin(); cliter != requestedcl.end(); ++cliter )
  {
    // check if myrank wants to bind this crosslinker
    bool myrankbond = false;
    if( mybonds.find(cliter->first) != mybonds.end() )
      myrankbond = true;

    // ---------------------------------------------------------------------
    // if only one request and myrank does not want to bind this cl,
    // requesting proc gets the permission to do so
    // ---------------------------------------------------------------------
    if(static_cast<int>(cliter->second.size()) == 1 and not myrankbond)
    {
      // we send back the permission to the relevant proc, because myrank as row
      // owner of bspot needs to set the respective stuff for the element of this
      // binding event
      // note: permission = true was send as default, so this can be sent back
      // without changes
      decidedbonds[cliter->second[0].requestproc].push_back(cliter->second[0]);

#ifdef DEBUG
      if ( cliter->second[0].permission != 1 )
        dserror(" something during communication went wrong, default true permission "
                " not received");
#endif

      // insert this new binding event in map of myrank, because as row owner of
      // this cl he is responsible to set the respective stuff for the crosslinker
      // of this binding event
      mybonds[cliter->first] = cliter->second[0];

      // go to next crosslinker
      continue;
    }

    // ---------------------------------------------------------------------
    // in case number of requesting procs >1 for this cl or myrank wants to
    // set it itself
    // ---------------------------------------------------------------------
    int numrequprocs = static_cast<int>( cliter->second.size() );
    if(myrankbond)
      numrequprocs += 1;

    // get random proc out of affected ones
    DRT::Problem::Instance()->Random()->SetRandRange( 0.0, 1.0 );
    // fixme: what if random number exactly = 1?
    int rankwithpermission = std::floor( numrequprocs * DRT::Problem::Instance()->Random()->Uni() );

    // myrank is allowed to set link
    if( myrankbond and rankwithpermission == (numrequprocs - 1) )
    {
      // note: this means link is set between row cl and row ele on myrank,
      // all relevant information for myrank is stored in mybonds
      // loop over all requesters and store their veto
      std::vector<BindEventData>::iterator iter;
      for( iter = cliter->second.begin(); iter != cliter->second.end(); ++iter )
      {
        iter->permission = 0;
        decidedbonds[iter->requestproc].push_back(*iter);
      }
    }
    // certain requester is allowed to set the link
    else
    {
      // loop over all requesters and store veto for all requester except for one
      std::vector<BindEventData>::iterator iter;

      int counter = 0;
      for( iter = cliter->second.begin(); iter != cliter->second.end(); ++iter )
      {
        if( rankwithpermission == counter )
        {
          // permission for this random proc
          decidedbonds[iter->requestproc].push_back(*iter);

#ifdef DEBUG
        if ( iter->permission != 1 )
          dserror(" something during communication went wrong, default true permission "
                  " not received");
#endif

          // erase old binding event
          if( myrankbond )
            mybonds.erase( cliter->first );

          // insert new binding event
          mybonds[cliter->first] = *iter;
        }
        else
        {
          iter->permission = 0;
          decidedbonds[iter->requestproc].push_back(*iter);
        }
        counter++;
      }
    }
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::CommunicateBeamToBeamLinkageAfterRedistribution(
    std::map<int, std::vector<Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> > >& dbondcltosend)
{
  CheckInit();

  // build exporter
  DRT::Exporter exporter(BinDiscret().Comm());

  // -----------------------------------------------------------------------
  // send
  // -----------------------------------------------------------------------
  // number of messages
  const int length = dbondcltosend.size();
  std::vector<MPI_Request> request(length);
  int tag = 0;
  std::map<int, std::vector<Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> > >::const_iterator p;
  for(p=dbondcltosend.begin(); p!=dbondcltosend.end(); ++p)
  {
    // ---- pack data for sending -----
    std::vector<char> sdata;
    std::vector<Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> >::const_iterator iter;
    DRT::PackBuffer data;
    for(iter=p->second.begin(); iter!=p->second.end(); ++iter)
    {
     (*iter)->Pack(data);
    }
    data.StartPacking();
    for(iter=p->second.begin(); iter!=p->second.end(); ++iter)
    {
      (*iter)->Pack(data);
    }
    swap(sdata,data());
     // unblocking send
    exporter.ISend(GState().GetMyRank(), p->first, &(sdata[0]), static_cast<int>(sdata.size()), 1234, request[tag]);
    ++tag;
  }
  if (tag != length) dserror("Number of messages is mixed up");

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  std::vector<int> summedtargets;
  PrepareReceivingProcs(dbondcltosend, summedtargets);

  // myrank receive all packs that are sent to him
  for(int rec=0; rec<summedtargets[GState().GetMyRank()]; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.ReceiveAny(from,tag,rdata,length);
    if (tag != 1234)
      dserror("Received on proc %i data with wrong tag from proc %i", GState().GetMyRank(), from);

    // store received data
    std::vector<char>::size_type position = 0;
    while (position < rdata.size())
    {
      std::vector<char> data;
      DRT::ParObject::ExtractfromPack(position,rdata,data);
      // this Teuchos::rcp holds the memory of the node
      Teuchos::RCP<DRT::ParObject> object = Teuchos::rcp(DRT::UTILS::Factory(data),true);
      Teuchos::RCP<BEAMINTERACTION::BeamToBeamLinkage> beamtobeamlink =
          Teuchos::rcp_dynamic_cast<BEAMINTERACTION::BeamToBeamLinkage>(object);
      if (beamtobeamlink == Teuchos::null) dserror("Received object is not a beam to beam linkage");

      // insert new double bonds in my list
      doublebondcl_[beamtobeamlink->Id()] = beamtobeamlink;
    }

    if (position != rdata.size())
      dserror("Mismatch in size of data %d <-> %d",static_cast<int>(rdata.size()),position);
  }

  // wait for all communication to finish
  Wait(exporter,request,static_cast<int>(dbondcltosend.size()));

}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::CommunicateCrosslinkerUnbinding(
    std::map<int, std::vector<UnBindEventData> >& sendunbindevent,
    std::vector<UnBindEventData>&                 myrankunbindevent) const
{
  CheckInit();

  ISendRecvAny( sendunbindevent, myrankunbindevent );

}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template<typename T>
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISend(
    DRT::Exporter& exporter,
    std::vector<MPI_Request>& request,
    const std::map<int, std::vector<T> >& send) const
{
  CheckInit();

  // number of messages
  const int length = static_cast<int>(send.size());
  request.resize(length);
  int tag = 0;
  typename std::map<int, std::vector<T> >::const_iterator p;
  for( p = send.begin(); p != send.end(); ++p )
  {
    // ---- pack data for sending -----
    std::vector<char> sdata;
    typename std::vector<T>::const_iterator iter;
    DRT::PackBuffer data;
    for( iter = p->second.begin(); iter != p->second.end(); ++iter )
    {
     Pack(data,*iter);
    }
    data.StartPacking();
    for( iter = p->second.begin(); iter != p->second.end(); ++iter )
    {
     Pack(data,*iter);
    }
    swap(sdata,data());

    // unblocking send
    exporter.ISend(GState().GetMyRank(), p->first, &(sdata[0]), static_cast<int>(sdata.size()), 1234, request[tag]);
    ++tag;
  }
  if (tag != length) dserror("Number of messages is mixed up");
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template<typename T>
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PrepareReceivingProcs(
    const std::map<int, std::vector<T> >& datasenttorank,
    std::vector<int>& summedtargets) const
{
  CheckInit();

  const int numproc = Discret().Comm().NumProc();

  // get number of procs from which myrank receives data
  std::vector<int> targetprocs(numproc,0);
  typename std::map<int, std::vector<T> >::const_iterator prociter;
  for(prociter=datasenttorank.begin(); prociter!=datasenttorank.end(); ++prociter)
    targetprocs[prociter->first] = 1;
  // store number of messages myrank receives
  summedtargets.resize(numproc,0);
  BinDiscret().Comm().SumAll(targetprocs.data(), summedtargets.data(), numproc);

}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template<typename T>
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::RecvAny(
    DRT::Exporter&  exporter,
    const int& receivesize,
    std::vector<T>& recv) const
{
  CheckInit();

  // myrank receive all packs that are sent to him
  for(int rec=0; rec<receivesize; ++rec)
  {
    std::vector<char> rdata;
    int length = 0;
    int tag = -1;
    int from = -1;
    exporter.ReceiveAny(from,tag,rdata,length);
    if (tag != 1234)
      dserror("Received on proc %i data with wrong tag from proc %i", GState().GetMyRank(), from);

    // store received data
    std::vector<char>::size_type position = 0;
    while (position < rdata.size())
    {
      // ---- extract received data -----
      T recdata;
      UnPack(position,rdata,recdata);

      // add received data to list of unbindevents on myrank
      recv.push_back(recdata);
    }

    if (position != rdata.size())
      dserror("Mismatch in size of data %d <-> %d",static_cast<int>(rdata.size()),position);
  }
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template<typename T>
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISendRecvAny(
  const std::map<int, std::vector<T> >& send,
  std::vector<T>&                       recv) const
{
  CheckInit();

  // build exporter
  DRT::Exporter exporter(BinDiscret().Comm());

  // -----------------------------------------------------------------------
  // send
  // -----------------------------------------------------------------------
  // unblocking send
  std::vector<MPI_Request> request;
  ISend(exporter,request,send);

  // -----------------------------------------------------------------------
  // prepare receive
  // -----------------------------------------------------------------------
  std::vector<int> summedtargets;
  PrepareReceivingProcs(send, summedtargets);

  // -----------------------------------------------------------------------
  // receive
  // -----------------------------------------------------------------------
  int receivesize = summedtargets[GState().GetMyRank()];
  RecvAny(exporter,receivesize,recv);

  // wait for all communication to finish
  Wait(exporter,request,static_cast<int>(send.size()));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Wait(
    DRT::Exporter& exporter,
    std::vector<MPI_Request>& request,
    const int& length) const
{
  CheckInit();

  // wait for all communication to finish
  for ( int i = 0; i < length; ++i )
    exporter.Wait(request[i]);

  // note: if we have done everything correct, this should be a no time operation
  BinDiscret().Comm().Barrier(); // I feel better this way ;-)
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Pack(
  DRT::PackBuffer&     data,
  const BindEventData& bindeventdata) const
{
  CheckInit();

  // pack data that is communicated
  DRT::ParObject::AddtoPack(data,bindeventdata.clgid);
  DRT::ParObject::AddtoPack(data,bindeventdata.elegid);
  DRT::ParObject::AddtoPack(data,bindeventdata.bspotlocn);
  DRT::ParObject::AddtoPack(data,bindeventdata.requestproc);
  DRT::ParObject::AddtoPack(data,bindeventdata.permission);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::Pack(
  DRT::PackBuffer&       data,
  const UnBindEventData& unbindeventdata) const
{
  CheckInit();

  // pack data that is communicated
  DRT::ParObject::AddtoPack(data,unbindeventdata.eletoupdate);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UnPack(
  std::vector<char>::size_type& position,
  std::vector<char>             data,
  BindEventData&                bindeventdata) const
{
  CheckInit();

  // extract data
  DRT::ParObject::ExtractfromPack(position,data,bindeventdata.clgid);
  DRT::ParObject::ExtractfromPack(position,data,bindeventdata.elegid);
  DRT::ParObject::ExtractfromPack(position,data,bindeventdata.bspotlocn);
  DRT::ParObject::ExtractfromPack(position,data,bindeventdata.requestproc);
  DRT::ParObject::ExtractfromPack(position,data,bindeventdata.permission);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::UnPack(
  std::vector<char>::size_type& position,
  std::vector<char>             data,
  UnBindEventData&              unbindeventdata) const
{
  CheckInit();

  // extract data
  DRT::ParObject::ExtractfromPack(position,data,unbindeventdata.eletoupdate);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PrintAndCheckBineEventData(
  BindEventData const& bindeventdata) const
{
  CheckInit();

  // extract data
  std::cout << "\n Rank: " << GState().GetMyRank() << std::endl;
  std::cout << " crosslinker gid " << bindeventdata.clgid << std::endl;
  std::cout << " element gid " << bindeventdata.elegid << std::endl;
  std::cout << " bspot local number " << bindeventdata.bspotlocn << std::endl;
  std::cout << " requesting proc " << bindeventdata.requestproc << std::endl;
  std::cout << " permission " << bindeventdata.permission << std::endl;

  if( bindeventdata.clgid < 0 or bindeventdata.elegid < 0 or bindeventdata.bspotlocn < 0
      or bindeventdata.requestproc < 0 or not (bindeventdata.permission == 0 or bindeventdata.permission == 1 ) )
    dserror(" your bindevent does not make sense.");
}


//-----------------------------------------------------------------------------
// explicit template instantiation (to please every compiler)
//-----------------------------------------------------------------------------
template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISend(
    DRT::Exporter&,std::vector<MPI_Request>&,const std::map<int, std::vector<BindEventData> >&) const;
template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISend(
    DRT::Exporter&,std::vector<MPI_Request>&,const std::map<int, std::vector<UnBindEventData> >&) const;

template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PrepareReceivingProcs(
    const std::map<int, std::vector<BindEventData> >&,std::vector<int>&) const;
template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::PrepareReceivingProcs(
    const std::map<int, std::vector<UnBindEventData> >&,std::vector<int>&) const;

template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::RecvAny(
    DRT::Exporter&,const int&,std::vector<BindEventData>&) const;
template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::RecvAny(
    DRT::Exporter&,const int&,std::vector<UnBindEventData>&) const;

template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISendRecvAny(
    const std::map<int, std::vector<BindEventData> >&,std::vector<BindEventData>&) const;
template void BEAMINTERACTION::SUBMODELEVALUATOR::Crosslinking::ISendRecvAny(
    const std::map<int, std::vector<UnBindEventData> >&,std::vector<UnBindEventData>&) const;