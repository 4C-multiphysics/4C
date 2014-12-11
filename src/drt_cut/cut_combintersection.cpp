/*!-----------------------------------------------------------------------------------------------*
\file cut_levelsetintersection.cpp

\brief provides the basic functionality for cutting a mesh with a level set function

<pre>
Maintainer: Benedikt Schott and Magnus Winter
            schott@lnm.mw.tum.de, winter@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15241
</pre>
 *------------------------------------------------------------------------------------------------*/
#include <Teuchos_TimeMonitor.hpp>

#include "cut_levelsetside.H"
#include "cut_levelsetintersection.H"
#include "cut_meshintersection.H"

#include "cut_combintersection.H"

/*-----------------------------------------------------------------------------------------*
 * constructur for Combined Intersection class (Levelset and Mesh intersection in one class)
 *-----------------------------------------------------------------------------------------*/

GEO::CUT::CombIntersection::CombIntersection(int myrank)
:   LevelSetIntersection( myrank, false), MeshIntersection(1, myrank), myrank_(myrank)
{
}


void GEO::CUT::CombIntersection::Cut(bool screenoutput)
{
  TEUCHOS_FUNC_TIME_MONITOR( "GEO::CUT --- 4/6 --- Cut_Intersection" );

  if(myrank_==0 and screenoutput) IO::cout << "\n\t * 4/6 Cut_Intersection ...";

  Mesh & m = NormalMesh();

  // Remark: we assume that there is no overlap between levelset-isocontour and mesh

  // find cut points with levelset-side
  if( side_ != Teuchos::null )
  {
    m.Cut( *side_ );
  }

  // find cut points with cut mesh
  m.FindCutPoints();

  m.MakeCutLines();
  m.MakeFacets();
  m.MakeVolumeCells();
}


void GEO::CUT::CombIntersection::FindNodePositions()
{
  // TODO: this function and the overall inside-outside position strategy still has to be adapted for more complex cases

  // NOTE: this will only work if mesh-cut area and level-set cut area are not overlapping

  Mesh & m = NormalMesh();

  // first, set the position for the mesh cut
  m.FindNodePositions();

  // second, set the position for the level-set cut
  m.FindLSNodePositions();

}


void GEO::CUT::CombIntersection::AddElement(
    int eid,
    const std::vector<int> & nids,
    const Epetra_SerialDenseMatrix & xyz,
    DRT::Element::DiscretizationType distype,
    const double * lsv,
    const bool lsv_only_plus_domain
)
{
  GEO::CUT::ElementHandle * e = NULL;

  // consider level-set values to decide whether the element has to be added or not
  if(lsv != NULL)
  {
    // NOTE: dependent on whether one or two phases are used for the computation, the number of degrees of freedom is determined via the
    // cut status of elements,
    // if both fluid phases have to be considered, we have to add only cut elements, as uncut elements always carry physical degrees of freedom
    // if only the plus domain is a physical field, we have to add also elements with pure negative level-set values (nodes in the ghost-domain)
    // such that the Dofset-Management does not produce degrees of freedom for such nodes

    e = LevelSetIntersection::AddElement(eid, nids, xyz, distype, lsv, lsv_only_plus_domain);
  }

  // no check necessary if element lies within bounding box of cut surface
  if(e!=NULL) return;

  MeshIntersection::AddElement(eid, nids, xyz, distype, lsv);

}

void GEO::CUT::CombIntersection::AddLevelSetSide(int levelset_side)
{
  LevelSetIntersection::AddCutSide(levelset_side);
}

void GEO::CUT::CombIntersection::AddMeshCuttingSide(
    int sid,
    const std::vector<int> & nids,
    const Epetra_SerialDenseMatrix & xyz,
    DRT::Element::DiscretizationType distype,
    int mi
    )
{
  MeshIntersection::AddCutSide(sid, nids, xyz, distype, mi);
}
