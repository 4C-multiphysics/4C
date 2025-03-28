// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_cut_tetmesh.hpp"

#include "4C_cut_levelsetside.hpp"
#include "4C_cut_tetmeshintersection.hpp"
#include "4C_cut_volumecell.hpp"

#include <stack>

extern "C"
{
#include <libqhull/qhull_a.h>
}

FOUR_C_NAMESPACE_OPEN


namespace Cut
{
  namespace
  {
    class NullFile
    {
     public:
      NullFile() : f_(nullptr) {}

      ~NullFile()
      {
        if (f_ != nullptr) fclose(f_);
      }

      operator FILE*()
      {
        if (f_ == nullptr) f_ = fopen("/dev/null", "w");
        return f_;
      }

     private:
      FILE* f_;
    };
  }  // namespace
}  // namespace Cut



Cut::TetMesh::TetMesh(
    const std::vector<Point*>& points, const plain_facet_set& facets, bool project)
    : points_(points), facets_(facets)
{
  std::vector<std::vector<int>> original_tets;

  call_q_hull(points_, original_tets, project);

  tets_.reserve(original_tets.size());
  for (std::vector<std::vector<int>>::iterator i = original_tets.begin(); i != original_tets.end();
      ++i)
  {
    std::vector<int>& t = *i;
    std::vector<Point*> tet;
    tet.reserve(t.size());
    for (std::vector<int>::iterator i = t.begin(); i != t.end(); ++i)
    {
      tet.push_back(points_[*i]);
    }
    if (is_valid_tet(tet))
    {
      tets_.push_back(t);
    }
  }

  init();
}

/* Loop over the facets of the tetmesh. Can every facet (triangulated at this stage) be associated
   with an unique TET from the delauney-cell decomposition?

   @Magnus:
   WARNING: For a LevelSet-cut with many cut-points this might lead to a failure to fill the
   FacetMesh as no info about the triangulated point is provided. This is however taken care of in
   the TetMeshIntersection, (the TETs are not cut) It could however be dealt with if the
   triangulated point is added to the cut information. Might have to look into this....

*/
bool Cut::TetMesh::fill_facet_mesh()
{
  for (plain_facet_set::const_iterator i = facets_.begin(); i != facets_.end(); ++i)
  {
    Facet* f = *i;
    if (not fill_facet(f))  // Possible to fill_facet?
    {
      return false;
    }
    if (f->has_holes())  // Does Facet contain holes?
    {
      const plain_facet_set& holes = f->holes();
      for (plain_facet_set::const_iterator i = holes.begin(); i != holes.end(); ++i)
      {
        Facet* h = *i;
        if (not fill_facet(h))
        {
          return false;
        }
      }
    }
  }
  return true;
}


/* From the input create TETs.
   1) First remove the TETs which are too small for arithmetic precision from the
   "accepted_tet_set" of TETs. However they might be needed later (for boundary-cell creation) so
   don't throw them away. 2) Try to "fill_facet_mesh(...)", this means that for the given given
   element (either a tet or a hex8) find the facets on the boundaries and find if they are
   associated with an unique TET.

    2.1) If NOT assume the tesselation (i.e. on TET or more) is cut, and call the Tesselation
   recursively. 2.2) If YES fill the inner part of the element with accepted tets and finish the
   tesselation for this element.

*/
void Cut::TetMesh::create_element_tets(Mesh& mesh, Element* element,
    const plain_volumecell_set& cells,
    const plain_side_set& cut_sides,  //<- cut_facets_ of parent ele.
    int count, bool tetcellsonly)
{
  fix_broken_tets();

  // Can the mesh be "filled from the facets"
  if (fill_facet_mesh())
  {
    for (plain_volumecell_set::const_iterator i = cells.begin(); i != cells.end(); ++i)
    {
      VolumeCell* vc = *i;

      // Describes the "domain" of the volume cell (i.e. its tets and borders)
      Domain<4> cell_domain;
      // The tets which are to be added in this volume cell
      PlainEntitySet<4>& cell_members = cell_domain.members();
      // The border (i.e. surface-tris) between the tets in the cell.
      PlainEntitySet<3>& cell_border = cell_domain.border();

      const plain_facet_set& facets = vc->facets();
      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        FacetMesh& fm = facet_mesh_[f];
        const PlainEntitySet<3>& tris = fm.surface_tris();

        // All tris are added to the cell_border, i.e. (border_ in class Domain).
        //  These are later used to create "done_border_" later in seed_domain.
        std::copy(tris.begin(), tris.end(), std::inserter(cell_border, cell_border.begin()));
      }

      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        seed_domain(cell_domain, f);
      }

      if (cell_domain.empty())
      {
        // Emergency. If this is the only volume cell within the element (an
        // element is surrounded by cut surfaces), try and force any available
        // tet into the volume cell. (This is a special case that turns up in
        // debug situations only.)

        if (cells.size() == 1)
        {
          for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
          {
            Facet* f = *i;
            seed_domain(cell_domain, f, true);
          }
        }

        if (cell_domain.empty())
        {
          // Assume the volume cell in question is degenerated and does not
          // contain any tets.

          // Test that the VC is completely on a cut-side. (added by Magnus)
          std::cout << "Warning a volume cell has no integration-cells!" << std::endl;

          bool volume_cell_on_facet = true;
          bool does_not_share_same_cutside = true;
          for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
          {
            Facet* f = *i;

            std::cout << "f->ShareSameCutSide(*facets.begin()): "
                      << f->share_same_cut_side(*facets.begin()) << std::endl;


            if (not f->share_same_cut_side(*facets.begin()))
            {
              std::cout << "NOT SAME ParentSide!" << std::endl;
              does_not_share_same_cutside = false;
              continue;
            }

            if (not f->is_planar(mesh, f->corner_points()))
            {
              std::cout << "Warning: This volume cell has no integration cells AND not all on same "
                           "cut-side."
                        << std::endl;
              volume_cell_on_facet = false;
              continue;
            }
          }

          if (not volume_cell_on_facet and not does_not_share_same_cutside)
            FOUR_C_THROW("cell_domain.Empty() and facets not on same cut-side and not planar");
          else if (not volume_cell_on_facet)
            FOUR_C_THROW("cell_domain.Empty() and facets not planar");
          else if (not does_not_share_same_cutside)
            FOUR_C_THROW("cell_domain.Empty() and facets not on same cut-side");

          continue;
        }
      }

      cell_domain.fill();

      std::vector<std::vector<Point*>> tets;
      tets.reserve(cell_members.size());

      for (PlainEntitySet<4>::iterator i = cell_members.begin(); i != cell_members.end(); ++i)
      {
        Entity<4>& t = **i;
        if (accept_tets_[t.id()])
        {
          std::vector<int>& fixedtet = tets_[t.id()];
          if (fixedtet.size() != 4) FOUR_C_THROW("confused");
          tets.push_back(std::vector<Point*>(4));
          std::vector<Point*>& tet = tets.back();
          for (int i = 0; i < 4; ++i)
          {
            tet[i] = points_[fixedtet[i]];
          }
        }
      }

      // all facets (whose facet-mesh are just tri3s here) which are on cut surface obtain the
      // coordinates to create boundary integration cells then
      std::map<Facet*, std::vector<Point*>> sides_xyz;

      for (plain_facet_set::const_iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        if (f->on_boundary_cell_side())  // This is entered for both sides of the volume cell.
        {
          FacetMesh& fm = facet_mesh_[f];
          const PlainEntitySet<3>& tris = fm.surface_tris();  // tris from the triangulated surface
          std::vector<Point*>& side_coords =
              sides_xyz[f];  // create entry for facet and get a reference to the facets side
                             // coordinates
          std::vector<std::vector<int>> sides;
          find_proper_sides(tris, sides, &cell_members);
          collect_coordinates(sides, side_coords);  // fill the side coordinates, if all the
                                                    // side's coordinates are on cut surface
        }
      }

      vc->create_tet4_integration_cells(mesh, tets, sides_xyz);

      if (vc->empty())
      {
        FOUR_C_THROW("empty volume cell detected");
      }
    }
  }
  else
  {
    // if ( count <= 3 )
    {
      TetMeshIntersection intersection(
          mesh.create_options(), element, tets_, accept_tets_, points_, cut_sides);
      intersection.cut(
          mesh, element, cells, count, tetcellsonly);  // writes tetmesh output in debug mode!!!
    }
  }
}

/* Initialize the data-structures needed for the TetMesh. The call to Qhull has to be performed
 * before.
 */
void Cut::TetMesh::init()
{
  unsigned numtets = tets_.size();

  for (unsigned i = 0; i < numtets; ++i)
  {
    const std::vector<int>& t = tets_[i];
    tet_entities_.push_back(Entity<4>(i, Handle<4>(t.data())));
  }

  for (std::vector<Entity<4>>::iterator i = tet_entities_.begin(); i != tet_entities_.end(); ++i)
  {
    Entity<4>& e = *i;
    e.create_children(tet_surfaces_);
  }

  for (std::map<Handle<3>, Entity<3>>::iterator i = tet_surfaces_.begin(); i != tet_surfaces_.end();
      ++i)
  {
    Entity<3>& e = i->second;
    e.create_children(tet_lines_);
  }

  accept_tets_.resize(tets_.size());
  std::fill(accept_tets_.begin(), accept_tets_.end(), true);

  //   active_surface_tris_.resize( tet_surfaces_.size() );
  //   std::fill( active_surface_tris_.begin(), active_surface_tris_.end(), false );
}


/* Call up QHull and let it create the delauney-cells. There are different options for how QHull
   can be called. 3 options are tested in order, if the previous option failed. Fill the
   std::vector tets with tetrahedrons provided from the delauney triangulization of the
   "convex"-hull.

   One might want to take a look at the call to Qhull to improve it for non-local coordinate cuts.
*/
void Cut::TetMesh::call_q_hull(
    const std::vector<Point*>& points, std::vector<std::vector<int>>& tets, bool project)
{
  const int dim = 3;
  const int n = points.size();

  if (n < 4)
  {
    FOUR_C_THROW("where coming from?");
    FOUR_C_THROW("illegal element topology");
  }
  //   if ( n == 4 )
  //   {
  //     FOUR_C_THROW( "no need to triangulate" );
  //   }

  std::vector<double> coordinates(dim * n);

  if (project)  // Never used... Not working properly either, I think...
  {
    Core::LinAlg::Matrix<3, 1> m;
    m = 0;
    double scale = 1. / n;
    for (int i = 0; i < n; ++i)  // Find mid-point (m)
    {
      Point* p = points[i];
      Core::LinAlg::Matrix<3, 1> x(p->x());
      m.update(scale, x, 1);
    }
    double length = 0;
    Core::LinAlg::Matrix<3, 1> l;
    for (int i = 0; i < n;
        ++i)  // Find the distance to the point furthest away from the mid-point (length)
    {
      Point* p = points[i];
      Core::LinAlg::Matrix<3, 1> x(p->x());
      l = m;
      l.update(1, x, -1);
      double n = l.norm2();
      length = std::max(n, length);
    }

    for (int i = 0; i < n; ++i)
    {
      Point* p = points[i];
      Core::LinAlg::Matrix<3, 1> x(p->x());
      l = m;
      l.update(1, x, -1);
      double n = l.norm2();
      l.scale(length / n);
      l.update(1, m, 1);
      std::copy(l.data(), l.data() + 3, &coordinates[dim * i]);
    }
  }
  else
  {
    for (int i = 0; i < n; ++i)
    {
      Point* p = points[i];
      p->coordinates(&coordinates[dim * i]);
    }
  }

  boolT ismalloc = false;

  // a set of option we try to process the input with
  // Qz seems to be required for rotational symmetric input
  std::vector<std::string> options;

  // Qhull options:
  //      Qbb - scale last coordinate to [0,m]. Preferable for integer input
  //      QbB - scales input to unit cube [-0.5,0.5]^3. Scaling can reduce precision errors if
  //      coord values wary widely.
#ifdef DIFF_QHULL_CALL
  options.push_back("qhull d Qt QbB Qc Pp");
#endif

  options.push_back("qhull d Qt Qbb Qc Pp");
  options.push_back("qhull d Qt Qbb Qc Qz Pp");
  options.push_back("qhull d Qt Qbb Qc QJ Pp");

  // If you want some debugging information replace the 0 pointer
  // with stdout or some other file open for writing.

#ifdef QHULL_EXTENDED_DEBUG_OUTPUT
  FILE* outfile = stdout;
#else
  FILE* outfile = nullptr;
#endif

#ifdef QHULL_DEBUG_OUTPUT
  FILE* errfile = stderr;
#else
  static NullFile errfile;
#endif

#ifdef QHULL_EXTENDED_DEBUG_OUTPUT
  int counter_qhull = 0;
#endif
  for (std::vector<std::string>::iterator i = options.begin(); i != options.end(); ++i)
  {
#ifdef QHULL_EXTENDED_DEBUG_OUTPUT
    std::cout << "counter_qhull: " << counter_qhull << std::endl;
#endif
    std::string& ostr = *i;
    if (not qh_new_qhull(dim, n, coordinates.data(), ismalloc, const_cast<char*>(ostr.c_str()),
            outfile, errfile))
    {
      // triangulate non-simplicial facets
      qh_triangulate();

      facetT* facet;
      int nf = 0;

      FORALLfacets
      {
        if (not facet->upperdelaunay) nf++;

        // Double check
        if (not facet->simplicial)
        {
          FOUR_C_THROW(
              "Qhull returned non-simplicial facets -- try delaunayn with different options");
        }
      }

      tets.reserve(nf);

      FORALLfacets
      {
        if (not facet->upperdelaunay)
        {
          if (facet->vertices)
          {
            std::vector<int> ids;
            ids.reserve(dim + 1);
            vertexT* vertex = nullptr;

            // FOREACHvertex_(facet->vertices)
            for (void** vertexp = &facet->vertices->e[0].p;
                (vertex = static_cast<vertexT*>(*vertexp++));)
            {
              int p = qh_pointid(vertex->point);
              if (p >= n)
              {
                FOUR_C_THROW("new node in delaunay");
              }
              ids.push_back(p);
            }

            tets.push_back(ids);
          }
        }
      }
    }

    // free long memory
    qh_freeqhull(not qh_ALL);

    // free short memory and memory allocator
    int curlong, totlong;
    qh_memfreeshort(&curlong, &totlong);

    if (curlong or totlong)
    {
      std::stringstream str;
      str << "did not free " << totlong << " bytes of long memory (" << curlong << " pieces)";
      FOUR_C_THROW("{}", str.str());
    }

    if (tets.size() > 0)
    {
      plain_int_set used_points;
      for (std::vector<std::vector<int>>::iterator i = tets.begin(); i != tets.end(); ++i)
      {
        std::vector<int>& t = *i;
        std::copy(t.begin(), t.end(), std::inserter(used_points, used_points.begin()));
      }
      if (used_points.size() == points.size())
      {
        return;
      }
      // FOUR_C_THROW( "failed to triangulate all points" );

      // failed! start a new iteration.
      tets.clear();
    }
#ifdef QHULL_EXTENDED_DEBUG_OUTPUT
    counter_qhull++;
#endif
  }

  // debug output to be read by qhull_test program
  FILE* debug_f = fopen("qhull.debug", "w");
  for (int i = 0; i < n; ++i)
  {
    Point* p = points[i];
    const double* x = p->x();
    fprintf(debug_f, "% .20f % .20f % .20f ", x[0], x[1], x[2]);
  }
  fprintf(debug_f, "\n");
  fclose(debug_f);

#ifdef QHULL_DEBUG_OUTPUT
  fflush(errfile);
#endif

  FOUR_C_THROW("qhull failed: Maybe the wrong version is used. Check your installation.");
}

/* First check if all the points of a tet share a cut-side, i.e. does the tet lie on a cut-side?
   If not, check if the points share a facet, if not ->         TET IS VALID.
   If yes, check if the cut_side is a LevelSetSide, if yes ->   TET IS VALID.
           if not LevelSetSide,
              check if the points share a facet, if no ->       TET IS VALID.
                   if yes,
                      check if facet is triangulated, if yes -> TET IS VALID.
                                                      if no  -> TET IS INVALID.

  + This was the way envisioned by Kuettler. However, it makes little sense accepting a TET
 completely on a cut-side. A simplification by me (Magnus) is to remove TETS whose points are all
 on a cut-side or more than one cut-side (A case not uncommon for MeshIntersection, however this
 works? Would it imply a TET is on a line?).
  + The change proposed does not change any existing test-case or any cut-test. However, the
 algorithm implemented, by Kuettler, is probably not there unnecessarily (hopefully?). Thus keep
 the code for reference in the future.
 !+ For now use the old way of Kuettler as after further investigation it seems to be somewhat
 more exact...

*/
bool Cut::TetMesh::is_valid_tet(const std::vector<Point*>& t)
{
  plain_side_set sides;
  // Find if the points of the tet share a common side.
  find_common_sides(t, sides);

  if (sides.size() == 0)
  {
    plain_facet_set facets;
    // Find if the points of the tet share a common facet.
    find_common_facets(t, facets);
    if (facets.size() == 0)
    {
      return true;
    }
    // Why do we have to enter here? Shouldn't it be clear already if the side=0, the points of
    // the tet
    //  can't share a common facet? A cut-side can be outside of the element, thus a case can
    //  occur, when a tet is not on a cut-side. BUT shares a facet. However this is already tested
    //  in FindCommonSides, as the points know what sides it cuts.
    FOUR_C_THROW(
        "You have encountered a case where the a tet is not on a cut-side BUT is on a facet!!! "
        "CHECK IT!");

    for (plain_facet_set::iterator i = facets.begin(); i != facets.end(); ++i)
    {
      Facet* f = *i;
      if (not f->is_triangulated())
      {
        return false;
      }
    }
    return true;
  }

  // Why would we want to accept a tet completely on a LevelSet-side?
  // POSSIBLE ANS: Might be a cut with more than 3nodes (i.e. 4,5,6) nodes NOT in a plane
  //               and tet defined by these 4 points?
  if (sides.size() == 1)
  {
    // Is this side a LevelSetSide, i.e. does the cut lie on a LevelSetSide?
    // The points of a tet can all be on a cut-side that is a LevelSetSide, but not share a common
    // facet. This occurs when there exists a degenerate cut (i.e. a cut we can't cut well). One
    // VC is created and that's about all.... Might have to investigate...
    if ((*sides.begin())->is_level_set_side())
    {
      // This part is done mainly for debugging purposes. It is probably not necessary,
      //  but could play an important role when/if the LevelSetSide is remodeled.
      plain_facet_set facets;
      // Find if the points of the tet share a common facet.
      find_common_facets(t, facets);

      // Eventhough all nodes share a level set side.
      if (facets.size() != 1)
      {
        //        std::cout << "facets.size(): " << facets.size();
        //        std::cout << "t.size(): " << t.size() << std::endl;
        //        for(unsigned k=0; k<t.size(); k++)
        //        {
        //          std::cout << "t[" << k << "]->Id()" << t[k]->Id() << std::endl;
        //          std::cout << "t[k]->CutSides().size(): " << t[k]->CutSides().size() <<
        //          std::endl; for(unsigned k2=0; k2<t[k]->CutSides().size(); k2++)
        //          {
        //            std::cout << "#"<< k2 << ": ";
        //            std::cout << "IsLevelSetSide: " << t[k]->CutSides()[k2]->IsLevelSetSide() <<
        //            std::endl; t[k]->CutSides()[k2]->print(); std::cout << std::endl;
        //          }
        //          for(unsigned k2=0; k2<t[k]->Facets().size(); k2++)
        //          {
        //            std::cout << "#"<< k2 << ": ";
        //            t[k]->Facets()[k2]->PrintPointIds();
        //            std::cout << std::endl;
        //          }
        //        }
        // FOUR_C_THROW("A LevelSetSide should BE associated to ONE facet. CHECK
        // THIS!");
      }

      for (plain_facet_set::iterator i = facets.begin(); i != facets.end(); ++i)
      {
        Facet* f = *i;
        if (f->is_triangulated())
        {
          return true;
        }
        else
          FOUR_C_THROW("Tet completely on LevelSetSide, and NOT Triangulated.");
      }
      return true;
    }
  }  // end if(sides.size()==1)

  // If a TET lies completely on a cut-side, remove this tet.
  // This is not the way of Kuettler!!! It is unclear what his vision for this was...
#ifdef REMOVE_ALL_TETS_ON_CUTSIDE
  if (sides.size() > 0)
  {
    return false;
  }
#else
  plain_facet_set facets;
  find_common_facets(t, facets);
  if (facets.size() == 0)
  {
    return true;
  }
  for (plain_facet_set::iterator i = facets.begin(); i != facets.end(); ++i)
  {
    Facet* f = *i;
    if (f->is_triangulated())  //(i.e. is this facet not a tri?)
    {
      return true;
    }
  }
#endif
  return false;
}

/* This function is unused....
 */
void Cut::TetMesh::test_used_points(const std::vector<std::vector<int>>& tets)
{
  plain_int_set used_points;
  for (std::vector<std::vector<int>>::const_iterator i = tets.begin(); i != tets.end(); ++i)
  {
    const std::vector<int>& t = *i;
    std::copy(t.begin(), t.end(), std::inserter(used_points, used_points.begin()));
  }
  if (used_points.size() != points_.size())
  {
    FOUR_C_THROW("failed to triangulate all points");
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Cut::TetMesh::fix_broken_tets()
{
  for (std::vector<std::vector<int>>::iterator i = tets_.begin(); i != tets_.end(); ++i)
  {
    std::vector<int>& t = *i;

    // create planes consisting of 3 nodes each
    Core::LinAlg::Matrix<3, 1> p0(points_[t[0]]->x());
    Core::LinAlg::Matrix<3, 1> p1(points_[t[1]]->x());
    Core::LinAlg::Matrix<3, 1> p2(points_[t[2]]->x());
    Core::LinAlg::Matrix<3, 1> p3(points_[t[3]]->x());

    Core::LinAlg::Matrix<3, 1> v01;
    Core::LinAlg::Matrix<3, 1> v02;
    Core::LinAlg::Matrix<3, 1> v03;

    v01.update(1, p1, -1, p0, 0);
    v02.update(1, p2, -1, p0, 0);
    v03.update(1, p3, -1, p0, 0);

    // create 4 normal vectors to each tet surface plane
    Core::LinAlg::Matrix<3, 1> nplane012;

    // cross product
    nplane012(0) = v01(1) * v02(2) - v01(2) * v02(1);
    nplane012(1) = v01(2) * v02(0) - v01(0) * v02(2);
    nplane012(2) = v01(0) * v02(1) - v01(1) * v02(0);

    // compute normal distance of point to plane of the three remaining points
    double distance = nplane012.dot(v03);

    // compute norm (area) of plane
    // double norm012 = nplane012.norm2();

    Core::LinAlg::Matrix<4, 1> temp(Core::LinAlg::Initialization::zero);
    temp(0, 0) = p0.norm2();  // Distance of points to origin
    temp(1, 0) = p1.norm2();
    temp(2, 0) = p2.norm2();
    temp(3, 0) = p3.norm2();

    // This is to scale the tolerance, it determines our maximum precision (i.e. machine
    // precision)
    double max_dist_to_origin = temp.norm_inf();

    Core::LinAlg::Matrix<3, 1> v04;
    v04.update(1, p1, -1, p2, 0);

    temp(0, 0) = v01.norm2();  // Distance between points in "base" triangle
    temp(1, 0) = v02.norm2();
    temp(2, 0) = v04.norm2();
    temp(3, 0) = fabs(distance);  //"Height" of tetrahedral

#ifdef NEW_POSTOL_TET
    // This is the smallest distance between the base points in the tet and the height.
    double min_dist_in_tet = temp.min_value();
    // We want to test with this one I think... But might lead to problems.
    double tolerance = LINSOLVETOL * max_dist_to_origin;
#else
    double vol_tet = distance / 6.0;
#endif

    // Deactivate all tets that are too small. We might still need the tet to
    // create a cut surface tri. Afterwards we will discard it.
#ifdef NEW_POSTOL_TET

    if (min_dist_in_tet < tolerance)
    {
      accept_tets_[i - tets_.begin()] = false;
    }
#else
    if (fabs(vol_tet) < VOLUMETOL)
    {
      accept_tets_[i - tets_.begin()] = false;
    }
    //     else if ( fabs( distance / norm012 ) < 1e-7 )
    //     {
    //       accept_tets_[i - tets_.begin()] = false;
    //     }
#endif

    // tet numbering wrong exchange 1 with 3
    if (distance < 0)
    {
      std::swap(t[1], t[3]);
    }
  }
}

/* Take the tri from the facet and test whether it belongs to more than one tet,
   if not add the tri as a side with an error check.
 */
void Cut::TetMesh::find_proper_sides(const PlainEntitySet<3>& tris,
    std::vector<std::vector<int>>& sides, const PlainEntitySet<4>* members)
{
  sides.reserve(tris.size());
  for (PlainEntitySet<3>::const_iterator i = tris.begin(); i != tris.end(); ++i)
  {
    Entity<3>* tri = *i;

    bool done = false;
    const std::vector<Entity<4>*>& tets = tri->parents();
    for (std::vector<Entity<4>*>::const_iterator i = tets.begin(); i != tets.end(); ++i)
    {
      Entity<4>* tet = *i;

      // Test if the tet exist as member in the volume-cell, if not skip this tet.
      if (members != nullptr and members->count(tet) == 0)
      {
        continue;
      }

      // Get the tet from the set of original tets.
      //  QUESTION: Why do we do this?
      //  Possible ans: Want access to the node-IDs.
      std::vector<int>& original_tet = tets_[tet->id()];

      if (original_tet.size() > 0)
      {
        if (done)
        {
          FOUR_C_THROW("double tets at cut surface");
        }

        done = true;

        bool found = false;
        for (int i = 0; i < 4; ++i)
        {
          std::vector<int> side(3);
          for (int j = 0; j < 3; ++j)
          {
            side[j] = original_tet[Core::FE::eleNodeNumbering_tet10_surfaces[i][j]];
          }
          if (tri->equals(side))
          {
            found = true;
            sides.push_back(std::vector<int>());
            std::swap(sides.back(), side);
            break;
          }
        }
        if (not found)
        {
          FOUR_C_THROW("failed to find side");
        }
      }
    }
    if (not done)
    {
      FOUR_C_THROW("failed to find tet");
    }
  }
}

/// Collects the coordinates for the tri3 sides of the facet if all its points are on cut surface
void Cut::TetMesh::collect_coordinates(
    const std::vector<std::vector<int>>& sides, std::vector<Point*>& side_coords)
{
  for (std::vector<std::vector<int>>::const_iterator i = sides.begin(); i != sides.end(); ++i)
  {
    const std::vector<int>& side = *i;

    Point* p1 = points_[side[0]];
    Point* p2 = points_[side[1]];
    Point* p3 = points_[side[2]];

    // Plausible: Might not need this check.
    //  However, left as is, as Tetmeshintersection is still not well understood.
    if (p1->has_associated_boundary_cell_facet() and p2->has_associated_boundary_cell_facet() and
        p3->has_associated_boundary_cell_facet())
    {
      side_coords.push_back(p1);
      side_coords.push_back(p2);
      side_coords.push_back(p3);
    }

    else
    {
      FOUR_C_THROW("Side not on cut or marked surface!!! Shouldn't it be?");
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
