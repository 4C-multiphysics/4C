/*----------------------------------------------------------------------------*/
/** \file

\brief XStructure/Structure state handling ( combination of XFEM and std.
       discretizations )



\level 3

*/
/*----------------------------------------------------------------------------*/

#include "xstr_xstructure_structure_state.H"

#include "xfield_state_utils.H"
#include "xfem_multi_field_mapextractor.H"
#include "xfem_dofset.H"

#include "linalg_utils_sparse_algebra_math.H"
#include "drt_discret_xfem.H"



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
XSTR::XStructureStructureState::XStructureStructureState()
{
  // intentionally left blank
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void XSTR::XStructureStructureState::Setup()
{
  CheckInit();
  XStructureState::Setup();
}
