/*----------------------------------------------------------------------*/
/*! \file
\brief Test for the CUT Library

\level 1

*----------------------------------------------------------------------*/

#include "4C_cut_element.hpp"
#include "4C_cut_mesh.hpp"
#include "4C_cut_options.hpp"
#include "4C_cut_triangulateFacet.hpp"

#include "cut_test_utils.hpp"

void check4nodedInline(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check4nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check5nodedInline(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check5nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check5nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check5nodedAdjacentconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check6nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check6nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check7nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check7nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check8nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check8nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check8nodedAdjacentconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check9nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check9nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check10nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check10nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check5nodedTwinConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check6nodedTwinConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check8nodedTriConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);
void check8nodedTriConcaveGenPlane(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check13nodedConvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check7nodedconti3concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check10nodedShift1ptConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check8nodedEarClip(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check7noded2concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void ProblemSplitAnyFacet1(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void InsideChcek1(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check15node5concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void check8nodeQuadInsidePt(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void checkTemporary(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

void checkTemporary2(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s);

Cut::Side* Create_quad4(Cut::Mesh& mesh, double x, double dx, double dz, bool reverse = false)
{
  Core::LinAlg::SerialDenseMatrix xyze(3, 4);

  xyze(0, 0) = x - dx;
  xyze(1, 0) = -0.5;
  xyze(2, 0) = -0.5 - dz;

  xyze(0, 1) = x + dx;
  xyze(1, 1) = -0.5;
  xyze(2, 1) = 1.5 + dz;

  xyze(0, 2) = x + dx;
  xyze(1, 2) = 1.5;
  xyze(2, 2) = 1.5 + dz;

  xyze(0, 3) = x - dx;
  xyze(1, 3) = 1.5;
  xyze(2, 3) = -0.5 - dz;

  if (reverse)
  {
    std::swap(xyze(0, 1), xyze(0, 3));
    std::swap(xyze(1, 1), xyze(1, 3));
    std::swap(xyze(2, 1), xyze(2, 3));
  }

  return create_quad4(mesh, xyze);
}

void test_facet_split()
{
  std::cout << "checking triangulation of facets...\n";
  Cut::Options options;
  options.init_for_cuttests();
  Cut::Mesh mesh(options);
  Cut::Element* e = create_hex8(mesh);
  Cut::Side* s = Create_quad4(mesh, 0.5, 0.1, 0);

  check4nodedInline(mesh, e, s);
  check4nodedconcave(mesh, e, s);
  check5nodedInline(mesh, e, s);
  check5nodedconvex(mesh, e, s);
  check5nodedconcave(mesh, e, s);
  check5nodedAdjacentconcave(mesh, e, s);
  check6nodedconvex(mesh, e, s);
  check6nodedconcave(mesh, e, s);
  check7nodedconvex(mesh, e, s);
  check7nodedconcave(mesh, e, s);
  check8nodedconvex(mesh, e, s);
  check8nodedconcave(mesh, e, s);
  check8nodedAdjacentconcave(mesh, e, s);
  check9nodedconvex(mesh, e, s);
  check9nodedconcave(mesh, e, s);
  check10nodedconvex(mesh, e, s);
  check10nodedconcave(mesh, e, s);
  check13nodedConvex(mesh, e, s);

  check5nodedTwinConcave(mesh, e, s);
  check6nodedTwinConcave(mesh, e, s);
  check8nodedTriConcave(mesh, e, s);
  check8nodedTriConcaveGenPlane(mesh, e, s);

  check7nodedconti3concave(mesh, e, s);
  check10nodedShift1ptConcave(mesh, e, s);
  check8nodedEarClip(mesh, e, s);
  check7noded2concave(mesh, e, s);
  ProblemSplitAnyFacet1(mesh, e, s);

  InsideChcek1(mesh, e, s);
  check15node5concave(mesh, e, s);
  check8nodeQuadInsidePt(mesh, e, s);

  /*checkTemporary( mesh, e, s );*/
  // checkTemporary2( mesh, e, s );
}

/*---------------------------------------------------------------------------------------*
 *      out of 4 nodes one should be deleted because it falls on the same line           *
 *---------------------------------------------------------------------------------------*/
void check4nodedInline(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check4nodedInline...\n";
  std::vector<Cut::Point*> ptlist(4);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.5;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p4)
    FOUR_C_THROW("triangulation failed for check4nodedInline");
}

/*---------------------------------------------------------------------------------------*
 *                A 4 noded concave facet is split into 2 triangles                      *
 *---------------------------------------------------------------------------------------*/
void check4nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check4nodedConcave...\n";
  std::vector<Cut::Point*> ptlist(4);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.5;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p2 || cell1[1] != p3 || cell1[2] != p4)
    FOUR_C_THROW("triangulation failed for check4nodedInline");
  if (cell2[0] != p2 || cell2[1] != p4 || cell2[2] != p1)
    FOUR_C_THROW("triangulation failed for check4nodedInline");
}

/*---------------------------------------------------------------------------------------*
 *        Out of 5 nodes, one inline node is deleted and a Quad cell is formed           *
 *---------------------------------------------------------------------------------------*/
void check5nodedInline(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check5nodedInline...\n";
  std::vector<Cut::Point*> ptlist(5);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.5;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.0;
  x[1] = 1.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p4 || cell1[3] != p5)
    FOUR_C_THROW("triangulation failed for check5nodedInline");
}

/*---------------------------------------------------------------------------------------*
 *         A 5 noded convex facet is split into a Tri and a Quad cell                    *
 *---------------------------------------------------------------------------------------*/
void check5nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check5nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(5);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.2;
  x[1] = 0.9;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.5;
  x[1] = 1.5;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check5nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5)
    FOUR_C_THROW("triangulation failed for check5nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *    A 5 noded facet with one concave point is split into a Tri and a Quad cell         *
 *---------------------------------------------------------------------------------------*/
void check5nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check5nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(5);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.2;
  x[1] = 0.9;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.5;
  x[1] = 0.7;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/


  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p1 || cell1[3] != p2)
    FOUR_C_THROW("triangulation failed for check5nodedconcave");
  if (cell2[0] != p4 || cell2[1] != p2 || cell2[2] != p3)
    FOUR_C_THROW("triangulation failed for check5nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 *          A 5 noded facet with 2 adjacent concave points --> 3 Tri cells               *
 *          This is test for EarClipping                                                 *
 *---------------------------------------------------------------------------------------*/
void check5nodedAdjacentconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check5nodedAdjacentconcave...\n";
  std::vector<Cut::Point*> ptlist(5);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.4;
  x[1] = 0.2;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.6;
  x[1] = 0.2;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
   }
  }*/


  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p5 || cell1[1] != p1 || cell1[2] != p2)
    FOUR_C_THROW("triangulation failed for check5nodedAdjacentconcave");
  if (cell2[0] != p3 || cell2[1] != p4 || cell2[2] != p5)
    FOUR_C_THROW("triangulation failed for check5nodedAdjacentconcave");
  if (cell3[0] != p3 || cell3[1] != p5 || cell3[2] != p2)
    FOUR_C_THROW("triangulation failed for check5nodedAdjacentconcave");
}

/*---------------------------------------------------------------------------------------*
 *                A 6 noded convex facet is split into 2 Quad cells                      *
 *---------------------------------------------------------------------------------------*/
void check6nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check6nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(6);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.2;
  x[1] = 0.9;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.5;
  x[1] = 1.5;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check6nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check6nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *          A 6 noded facet with one concave points is split into 2 Quad cells           *
 *---------------------------------------------------------------------------------------*/
void check6nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check6nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(6);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.2;
  x[1] = 0.9;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.5;
  x[1] = 0.7;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/


  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p5 || cell1[1] != p6 || cell1[2] != p1 || cell1[3] != p2)
    FOUR_C_THROW("triangulation failed for check6nodedconcave");
  if (cell2[0] != p5 || cell2[1] != p2 || cell2[2] != p3 || cell2[3] != p4)
    FOUR_C_THROW("triangulation failed for check6nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 *      7 noded facet with one concave points is split into 1 Tri and 2 Quad cells       *
 *---------------------------------------------------------------------------------------*/
void check7nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check7nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(7);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.4;
  x[1] = 0.6;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.1;
  x[1] = 0.9;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.3;
  x[1] = 1.5;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check7nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check7nodedconvex");
  if (cell3[0] != p1 || cell3[1] != p6 || cell3[2] != p7)
    FOUR_C_THROW("triangulation failed for check7nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *     A 6 noded facet with one concave points is split into 2 Quad and 1 Tri cells      *
 *---------------------------------------------------------------------------------------*/
void check7nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check7nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(7);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.2;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.4;
  x[1] = 0.6;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.1;
  x[1] = 0.9;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.3;
  x[1] = 0.7;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.1;
  x[1] = 1.0;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p6 || cell1[1] != p7 || cell1[2] != p1 || cell1[3] != p2)
    FOUR_C_THROW("triangulation failed for check7nodedconcave");
  if (cell2[0] != p6 || cell2[1] != p2 || cell2[2] != p3 || cell2[3] != p4)
    FOUR_C_THROW("triangulation failed for check7nodedconcave");
  if (cell3[0] != p6 || cell3[1] != p4 || cell3[2] != p5)
    FOUR_C_THROW("triangulation failed for check7nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 *                8 noded convex facet is split into 3 Quad cells                        *
 *---------------------------------------------------------------------------------------*/
void check8nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.5;
  x[1] = -0.4;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = -0.05;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.1;
  x[1] = 0.4;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.8;
  x[1] = 0.8;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = -0.1;
  x[1] = 0.1;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check8nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check8nodedconvex");
  if (cell3[0] != p1 || cell3[1] != p6 || cell3[2] != p7 || cell3[3] != p8)
    FOUR_C_THROW("triangulation failed for check8nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *       A 8 noded facet with one concave points is split into 3 Quad cells              *
 *---------------------------------------------------------------------------------------*/
void check8nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.5;
  x[1] = -0.4;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = -0.05;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.1;
  x[1] = 0.4;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.8;
  x[1] = 0.8;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.4;
  x[1] = 0.1;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p8 || cell1[1] != p1 || cell1[2] != p2 || cell1[3] != p3)
    FOUR_C_THROW("triangulation failed for check8nodedconcave");
  if (cell2[0] != p8 || cell2[1] != p3 || cell2[2] != p4 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedconcave");
  if (cell3[0] != p8 || cell3[1] != p5 || cell3[2] != p6 || cell3[3] != p7)
    FOUR_C_THROW("triangulation failed for check8nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 * A 8 noded facet with 2 adjacent concave points is split into 2 Quad and 2 Tri cells   *
 * First earclipping is called, and once adjacent concave pts are removed,               *
 * SplitAnyFacet is used to get Quad cells                                               *
 *---------------------------------------------------------------------------------------*/
void check8nodedAdjacentconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedAdjacentconcave...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 2.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 3.0;
  x[1] = 0.0;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 3.0;
  x[1] = 2.0;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.0;
  x[1] = 2.0;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6 || cell1[3] != p7)
    FOUR_C_THROW("triangulation failed for check8nodedAdjacentconcave");
  if (cell2[0] != p3 || cell2[1] != p4 || cell2[2] != p7 || cell2[3] != p8)
    FOUR_C_THROW("triangulation failed for check8nodedAdjacentconcave");
  if (cell3[0] != p3 || cell3[1] != p8 || cell3[2] != p1 || cell3[3] != p2)
    FOUR_C_THROW("triangulation failed for check8nodedAdjacentconcave");
}

/*---------------------------------------------------------------------------------------*
 *        9 noded convex facet is split into 3 Quad and 1 Tri cells                      *
 *---------------------------------------------------------------------------------------*/
void check9nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check9nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(9);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.5;
  x[1] = -0.4;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = -0.05;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.15;
  x[1] = 0.3;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 1.1;
  x[1] = 0.4;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.8;
  x[1] = 0.8;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = -0.1;
  x[1] = 0.1;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check9nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check9nodedconvex");
  if (cell3[0] != p1 || cell3[1] != p6 || cell3[2] != p7 || cell3[3] != p8)
    FOUR_C_THROW("triangulation failed for check9nodedconvex");
  if (cell4[0] != p1 || cell4[1] != p8 || cell4[2] != p9)
    FOUR_C_THROW("triangulation failed for check9nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *    A 9 noded facet with one concave points is split into 1 Tri and3 Quad cells        *
 *---------------------------------------------------------------------------------------*/
void check9nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check9nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(9);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.5;
  x[1] = -0.4;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = -0.05;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.15;
  x[1] = 0.3;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 1.1;
  x[1] = 0.4;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.8;
  x[1] = 0.8;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 0.1;
  x[1] = 0.1;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p9 || cell1[1] != p1 || cell1[2] != p2 || cell1[3] != p3)
    FOUR_C_THROW("triangulation failed for check9nodedconcave");
  if (cell2[0] != p9 || cell2[1] != p3 || cell2[2] != p4 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check9nodedconcave");
  if (cell3[0] != p9 || cell3[1] != p5 || cell3[2] != p6 || cell3[3] != p7)
    FOUR_C_THROW("triangulation failed for check9nodedconcave");
  if (cell4[0] != p9 || cell4[1] != p7 || cell4[2] != p8)
    FOUR_C_THROW("triangulation failed for check9nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 *              10 noded convex facet is split into 4 Quad cells                         *
 *---------------------------------------------------------------------------------------*/
void check10nodedconvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check10nodedconvex...\n";
  std::vector<Cut::Point*> ptlist(10);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.3;
  x[1] = 0.2;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 1.0;
  x[1] = 0.4;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.2;
  x[1] = 0.7;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 0.0;
  x[1] = 0.4;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  x[0] = -0.1;
  x[1] = 0.2;
  Cut::Point* p10 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[9] = p10;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p1 || cell1[1] != p2 || cell1[2] != p3 || cell1[3] != p4)
    FOUR_C_THROW("triangulation failed for check10nodedconvex");
  if (cell2[0] != p1 || cell2[1] != p4 || cell2[2] != p5 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check10nodedconvex");
  if (cell3[0] != p1 || cell3[1] != p6 || cell3[2] != p7 || cell3[3] != p8)
    FOUR_C_THROW("triangulation failed for check10nodedconvex");
  if (cell4[0] != p1 || cell4[1] != p8 || cell4[2] != p9 || cell4[3] != p10)
    FOUR_C_THROW("triangulation failed for check10nodedconvex");
}

/*---------------------------------------------------------------------------------------*
 *        A 10 noded facet with one concave points is split into 4 Quad cells            *
 *---------------------------------------------------------------------------------------*/
void check10nodedconcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check10nodedconcave...\n";
  std::vector<Cut::Point*> ptlist(10);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = -0.3;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.4;
  x[1] = -0.3;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.3;
  x[1] = 0.2;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 1.0;
  x[1] = 0.4;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.4;
  x[1] = 0.7;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.2;
  x[1] = 0.7;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 0.0;
  x[1] = 0.4;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  x[0] = 0.1;
  x[1] = 0.3;
  Cut::Point* p10 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[9] = p10;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p10 || cell1[1] != p1 || cell1[2] != p2 || cell1[3] != p3)
    FOUR_C_THROW("triangulation failed for check10nodedconcave");
  if (cell2[0] != p10 || cell2[1] != p3 || cell2[2] != p4 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check10nodedconcave");
  if (cell3[0] != p10 || cell3[1] != p5 || cell3[2] != p6 || cell3[3] != p7)
    FOUR_C_THROW("triangulation failed for check10nodedconcave");
  if (cell4[0] != p10 || cell4[1] != p7 || cell4[2] != p8 || cell4[3] != p9)
    FOUR_C_THROW("triangulation failed for check10nodedconcave");
}

/*---------------------------------------------------------------------------------------*
 *     A 5 noded facet with 2 (non-adjacent) concave pt is split into 3 Tri cells        *
 *     Check for SplitAnyFacet                                                           *
 *---------------------------------------------------------------------------------------*/
void check5nodedTwinConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check5nodedTwinConcave...\n";
  std::vector<Cut::Point*> ptlist(5);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.2;
  x[1] = 0.1;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.3;
  x[1] = 0.2;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.2;
  x[1] = 0.7;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }
  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p5 || cell1[1] != p1 || cell1[2] != p2)
    FOUR_C_THROW("triangulation failed for check5nodedTwinConcave");
  if (cell2[0] != p4 || cell2[1] != p5 || cell2[2] != p2)
    FOUR_C_THROW("triangulation failed for check5nodedTwinConcave");
  if (cell3[0] != p4 || cell3[1] != p2 || cell3[2] != p3)
    FOUR_C_THROW("triangulation failed for check5nodedTwinConcave");
}

/*---------------------------------------------------------------------------------------*
 *     A 6 noded special facet with 2 (non-adjacent) concave pt is split into            *
 *     2 Quad cells  -- Check for SplitAnyFacet                                          *
 *---------------------------------------------------------------------------------------*/
void check6nodedTwinConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check6nodedTwinConcave...\n";
  std::vector<Cut::Point*> ptlist(6);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.3;
  x[1] = 0.7;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 2.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.0;
  x[1] = 1.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.9;
  x[1] = 0.7;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  if (cell1[0] != p3 || cell1[1] != p4 || cell1[2] != p5 || cell1[3] != p6)
    FOUR_C_THROW("triangulation failed for check6nodedTwinConcave");
  if (cell2[0] != p1 || cell2[1] != p2 || cell2[2] != p3 || cell2[3] != p6)
    FOUR_C_THROW("triangulation failed for check6nodedTwinConcave");
}

/*---------------------------------------------------------------------------------------*
 *     A 8 noded facet with 3 (no non-adjacent) concave pts is split into                *
 *     3 Quad cells  -- Check for SplitAnyFacet                                          *
 *---------------------------------------------------------------------------------------*/
void check8nodedTriConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedTriConcave...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.5;
  x[1] = 0.2;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.5;
  x[1] = 0.7;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.0;
  x[1] = 1.0;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.0;
  x[1] = 0.8;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.2;
  x[1] = 0.5;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p2 || cell1[1] != p3 || cell1[2] != p4 || cell1[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcave");
  if (cell2[0] != p8 || cell2[1] != p1 || cell2[2] != p2 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcave");
  if (cell3[0] != p8 || cell3[1] != p5 || cell3[2] != p6 || cell3[3] != p7)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcave");
}

/*---------------------------------------------------------------------------------------*
 *     A 8 noded facet with 3 (no non-adjacent) concave pts is split into                *
 *     3 Quad cells  -- Check for SplitAnyFacet                                          *
 *     Except this all other tests contains facets that are in z=0 plane                 *
 *     facet in this example is in 2x+3y+4z=7 plane                                      *
 *---------------------------------------------------------------------------------------*/
void check8nodedTriConcaveGenPlane(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedTriConcaveGenPlane...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];

  x[1] = 0.0;
  x[2] = 0.0;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[1] = 0.5;
  x[2] = 0.2;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[1] = 1.0;
  x[2] = 0.0;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[1] = 1.0;
  x[2] = 1.0;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[1] = 0.5;
  x[2] = 0.7;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[1] = 0.0;
  x[2] = 1.0;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[1] = 0.0;
  x[2] = 0.8;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[1] = 0.2;
  x[2] = 0.5;
  x[0] = 0.5 * (7.0 - 3.0 * x[1] - 4.0 * x[2]);
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p2 || cell1[1] != p3 || cell1[2] != p4 || cell1[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcaveGenPlane");
  if (cell2[0] != p8 || cell2[1] != p1 || cell2[2] != p2 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcaveGenPlane");
  if (cell3[0] != p8 || cell3[1] != p5 || cell3[2] != p6 || cell3[3] != p7)
    FOUR_C_THROW("triangulation failed for check8nodedTriConcaveGenPlane");
}

/*---------------------------------------------------------------------------------------*
 *          13 noded convex facet is split into 5 Quad and a Tri cell                    *
 *---------------------------------------------------------------------------------------*/
void check13nodedConvex(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check13nodedConvex...\n";
  std::vector<Cut::Point*> ptlist(13);
  double x[3];
  x[2] = 0.0;

  // std::cout<<"the points are  \n";
  for (int i = 0; i < 13; i++)
  {
    double theta = 2.0 * 22.0 / 7.0 / 13 * i;
    x[0] = cos(theta);
    x[1] = sin(theta);

    Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
    ptlist[i] = p1;
  }

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  if (split.size() != 6) FOUR_C_THROW("triangulation failed for check13nodedConvex");

  for (int i = 0; i < 6; i++)
  {
    std::vector<Cut::Point*> cell = split[i];
    if (i == 5)
    {
      if (cell[0] != ptlist[0] || cell[1] != ptlist[i * 2 + 1] || cell[2] != ptlist[i * 2 + 2])
        FOUR_C_THROW("triangulation failed for check13nodedConvex");
    }
    else
    {
      if (cell[0] != ptlist[0] || cell[1] != ptlist[i * 2 + 1] || cell[2] != ptlist[i * 2 + 2] ||
          cell[3] != ptlist[i * 2 + 3])
        FOUR_C_THROW("triangulation failed for check13nodedConvex");
    }
  }
}

/*---------------------------------------------------------------------------------------*
 *     7 noded facet with 3 continuous concave points is split into 5 Tri cells          *
 *     call split_general_facet first, and moves to EarClipping                            *
 *---------------------------------------------------------------------------------------*/
void check7nodedconti3concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check7nodedconti3concave...\n";
  std::vector<Cut::Point*> ptlist(7);
  double x[3];
  x[2] = 0.0;

  x[0] = 1.0;
  x[1] = 0.728885;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.992803;
  x[1] = 0.726274;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.958516;
  x[1] = 0.710801;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.950656;
  x[1] = 0.706818;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.916667;
  x[1] = 0.671407;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.916667;
  x[1] = 0.75;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 1.0;
  x[1] = 0.75;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  std::vector<Cut::Point*> cell5 = split[4];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6)
    FOUR_C_THROW("triangulation failed for check7nodedconti3concave");
  if (cell2[0] != p4 || cell2[1] != p6 || cell2[2] != p7)
    FOUR_C_THROW("triangulation failed for check7nodedconti3concave");
  if (cell3[0] != p7 || cell3[1] != p1 || cell3[2] != p2)
    FOUR_C_THROW("triangulation failed for check7nodedconti3concave");
  if (cell4[0] != p3 || cell4[1] != p4 || cell4[2] != p7)
    FOUR_C_THROW("triangulation failed for check7nodedconti3concave");
  if (cell5[0] != p3 || cell5[1] != p7 || cell5[2] != p2)
    FOUR_C_THROW("triangulation failed for check7nodedconti3concave");
}

/*---------------------------------------------------------------------------------------*
 *    A 10 noded facet: first call SplitGeneralFafet, then move to 1ptconcave split      *
 *---------------------------------------------------------------------------------------*/
void check10nodedShift1ptConcave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check10nodedShiftEarClipToSplit...\n";
  std::vector<Cut::Point*> ptlist(10);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 2.0;
  x[1] = 1.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 3.0;
  x[1] = 0.0;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 2.5;
  x[1] = 1.0;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 3.0;
  x[1] = 2.0;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 1.5;
  x[1] = 1.5;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  x[0] = 0.0;
  x[1] = 2.0;
  Cut::Point* p10 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[9] = p10;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6 || cell1[3] != p7)
    FOUR_C_THROW("triangulation failed for check10nodedShiftEarClipToSplit");
  if (cell2[0] != p3 || cell2[1] != p7 || cell2[2] != p8 || cell2[3] != p9)
    FOUR_C_THROW("triangulation failed for check10nodedShiftEarClipToSplit");
  if (cell3[0] != p9 || cell3[1] != p10 || cell3[2] != p1)
    FOUR_C_THROW("triangulation failed for check10nodedShiftEarClipToSplit");
  if (cell4[0] != p3 || cell4[1] != p1 || cell4[2] != p2)
    FOUR_C_THROW("triangulation failed for check10nodedShiftEarClipToSplit");
}

/*---------------------------------------------------------------------------------------*
 *        Check for EarClipping -- 8 noded facet is split to yield 6 triangles           *
 *---------------------------------------------------------------------------------------*/
void check8nodedEarClip(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodedEarClip...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[1] = 0.426667;

  x[0] = 0.888889;
  x[2] = 0.333333;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.944444;
  x[2] = 0.333333;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.944444;
  x[2] = 0.291667;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.942495;
  x[2] = 0.291667;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.942497;
  x[2] = 0.320888;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.916262;
  x[2] = 0.320832;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.916332;
  x[2] = 0.291667;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.888889;
  x[2] = 0.291667;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);

  std::vector<int> ptc;
  tf.ear_clipping(ptc, true);

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  std::vector<Cut::Point*> cell5 = split[4];
  std::vector<Cut::Point*> cell6 = split[5];
  if (cell1[0] != p2 || cell1[1] != p3 || cell1[2] != p4)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
  if (cell2[0] != p2 || cell2[1] != p4 || cell2[2] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
  if (cell3[0] != p1 || cell3[1] != p2 || cell3[2] != p5)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
  if (cell4[0] != p1 || cell4[1] != p5 || cell4[2] != p6)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
  if (cell5[0] != p8 || cell5[1] != p1 || cell5[2] != p6)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
  if (cell6[0] != p6 || cell6[1] != p7 || cell6[2] != p8)
    FOUR_C_THROW("triangulation failed for check8nodedEarClip");
}

/*---------------------------------------------------------------------------------------*
 *           A noded 2 continuous concave pts facet split into 5 Tri                     *
 *---------------------------------------------------------------------------------------*/
void check7noded2concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check7nodedShiftSplitToEarClip...\n";
  std::vector<Cut::Point*> ptlist(7);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.0;
  x[1] = 1.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 0.5;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.75;
  x[1] = 0.8;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.5;
  x[1] = 0.3;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.25;
  x[1] = 0.5;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  std::vector<Cut::Point*> cell5 = split[4];
  if (cell1[0] != p5 || cell1[1] != p6 || cell1[2] != p7)
    FOUR_C_THROW("triangulation failed for check7nodedShiftSplitToEarClip");
  if (cell2[0] != p7 || cell2[1] != p1 || cell2[2] != p2)
    FOUR_C_THROW("triangulation failed for check7nodedShiftSplitToEarClip");
  if (cell3[0] != p7 || cell3[1] != p2 || cell3[2] != p3)
    FOUR_C_THROW("triangulation failed for check7nodedShiftSplitToEarClip");
  if (cell4[0] != p5 || cell4[1] != p7 || cell4[2] != p3)
    FOUR_C_THROW("triangulation failed for check7nodedShiftSplitToEarClip");
  if (cell5[0] != p5 || cell5[1] != p3 || cell5[2] != p4)
    FOUR_C_THROW("triangulation failed for check7nodedShiftSplitToEarClip");
}

/*---------------------------------------------------------------------------------------*
 * 6 noded facet with 2 concave points --- old SplitAnyFacet produced intersecting cells *
 *---------------------------------------------------------------------------------------*/
void ProblemSplitAnyFacet1(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "ProblemSplitAnyFacet...\n";
  std::vector<Cut::Point*> ptlist(6);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.0;
  x[1] = 1.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 1.0;
  x[1] = 1.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.8;
  x[1] = 0.95;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.8;
  x[1] = 0.2;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.5;
  x[1] = 0.5;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6)
    FOUR_C_THROW("triangulation failed for ProblemSplitAnyFacet1");
  if (cell2[0] != p4 || cell2[1] != p6 || cell2[2] != p1 || cell2[3] != p2)
    FOUR_C_THROW("triangulation failed for ProblemSplitAnyFacet1");
  if (cell3[0] != p4 || cell3[1] != p2 || cell3[2] != p3)
    FOUR_C_THROW("triangulation failed for ProblemSplitAnyFacet1");
}

/*---------------------------------------------------------------------------------------*
 *     A 7 noded facet that make sure that inside checking of points is necessary        *
 *---------------------------------------------------------------------------------------*/
void InsideChcek1(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "InsideChcek1...\n";
  std::vector<Cut::Point*> ptlist(7);
  double x[3];
  x[2] = 0.0;

  x[0] = -0.5;
  x[1] = -0.5;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = -0.5;
  x[1] = 0.5;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 0.5;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.0;
  x[1] = -0.5;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.5;
  x[1] = -0.5;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.5;
  x[1] = 0.0;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6 || cell1[3] != p7)
    FOUR_C_THROW("triangulation failed for InsideChcek1");
  if (cell2[0] != p3 || cell2[1] != p4 || cell2[2] != p7 || cell2[3] != p1)
    FOUR_C_THROW("triangulation failed for InsideChcek1");
  if (cell3[0] != p3 || cell3[1] != p1 || cell3[2] != p2)
    FOUR_C_THROW("triangulation failed for InsideChcek1");
}

/*---------------------------------------------------------------------------------------*
 *                 A 15 noded facet with 5 concave points                                *
 *---------------------------------------------------------------------------------------*/
void check15node5concave(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check15node5concave...\n";
  std::vector<Cut::Point*> ptlist(15);
  double x[3];
  x[2] = 0.0;

  x[0] = 0.0;
  x[1] = 0.0;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.5;
  x[1] = 0.0;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.5;
  x[1] = -1.0;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 1.0;
  x[1] = 0.0;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 1.5;
  x[1] = 0.0;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 1.3;
  x[1] = -0.8;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 1.5;
  x[1] = -0.8;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 2.0;
  x[1] = -0.5;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 2.0;
  x[1] = 0.0;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  x[0] = 2.2;
  x[1] = 0.0;
  Cut::Point* p10 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[9] = p10;

  x[0] = 2.2;
  x[1] = -2.0;
  Cut::Point* p11 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[10] = p11;

  x[0] = 1.5;
  x[1] = -2.0;
  Cut::Point* p12 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[11] = p12;

  x[0] = 0.8;
  x[1] = -0.8;
  Cut::Point* p13 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[12] = p13;

  x[0] = 0.8;
  x[1] = -2.0;
  Cut::Point* p14 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[13] = p14;

  x[0] = 0.0;
  x[1] = -2.0;
  Cut::Point* p15 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[14] = p15;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  std::vector<Cut::Point*> cell5 = split[4];
  std::vector<Cut::Point*> cell6 = split[5];
  std::vector<Cut::Point*> cell7 = split[6];
  std::vector<Cut::Point*> cell8 = split[7];
  if (cell1[0] != p4 || cell1[1] != p5 || cell1[2] != p6)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell2[0] != p8 || cell2[1] != p9 || cell2[2] != p10 || cell2[3] != p11)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell3[0] != p7 || cell3[1] != p8 || cell3[2] != p11 || cell3[3] != p12)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell4[0] != p15 || cell4[1] != p1 || cell4[2] != p2 || cell4[3] != p3)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell5[0] != p13 || cell5[1] != p14 || cell5[2] != p15)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell6[0] != p13 || cell6[1] != p15 || cell6[2] != p4)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell7[0] != p7 || cell7[1] != p12 || cell7[2] != p13)
    FOUR_C_THROW("triangulation failed for check15node5concave");
  if (cell8[0] != p6 || cell8[1] != p13 || cell8[2] != p4)
    FOUR_C_THROW("triangulation failed for check15node5concave");
}

/*---------------------------------------------------------------------------------------*
 *                 A 8 noded facet -- check for QuadInsidePt                             *
 *---------------------------------------------------------------------------------------*/
void check8nodeQuadInsidePt(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "check8nodeQuadInsidePt...\n";
  std::vector<Cut::Point*> ptlist(8);
  double x[3];
  x[0] = 5.44;

  x[1] = 4.8;
  x[2] = 4.88;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[1] = 4.8;
  x[2] = 4.8;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[1] = 4.88;
  x[2] = 4.8;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[1] = 4.88;
  x[2] = 4.80242;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[1] = 4.85799;
  x[2] = 4.81544;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[1] = 4.83652;
  x[2] = 4.83652;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[1] = 4.81544;
  x[2] = 4.85799;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[1] = 4.80242;
  x[2] = 4.88;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  /*std::cout<<"the corner points of facet are\n";
  for( std::vector<Cut::Point*>::iterator i=ptlist.begin();i!=ptlist.end();i++ )
  {
    Cut::Point* pt = *i;
    double coo[3];
    pt->coordinates(coo);
    std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
  }

  for( std::vector<std::vector<Cut::Point*> >::iterator i=split.begin();i!=split.end();i++ )
  {
    std::cout<<"cell\n";
    std::vector<Cut::Point*> cell = *i;;
    for( std::vector<Cut::Point*>::iterator j=cell.begin();j!=cell.end();j++ )
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout<<coo[0]<<"\t"<<coo[1]<<"\t"<<coo[2]<<"\n";
    }
  }*/

  std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  if (cell1[0] != p7 || cell1[1] != p8 || cell1[2] != p1 || cell1[3] != p2)
    FOUR_C_THROW("triangulation failed for check8nodeQuadInsidePt");
  if (cell2[0] != p2 || cell2[1] != p3 || cell2[2] != p4 || cell2[3] != p5)
    FOUR_C_THROW("triangulation failed for check8nodeQuadInsidePt");
  if (cell3[0] != p6 || cell3[1] != p7 || cell3[2] != p2)
    FOUR_C_THROW("triangulation failed for check8nodeQuadInsidePt");
  if (cell4[0] != p6 || cell4[1] != p2 || cell4[2] != p5)
    FOUR_C_THROW("triangulation failed for check8nodeQuadInsidePt");
}

void checkTemporary(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "checkTemporary...\n";
  std::vector<Cut::Point*> ptlist(12);
  double x[3];

  x[0] = 0.88888889999999998182;
  x[1] = 0.42666670000000000984;
  x[2] = 0.33333329999999999904;
  Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[0] = p1;

  x[0] = 0.94444439999999996171;
  x[1] = 0.42666670000000000984;
  x[2] = 0.33333329999999999904;
  Cut::Point* p2 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[1] = p2;

  x[0] = 0.94444439999999996171;
  x[1] = 0.42666670000000000984;
  x[2] = 0.29166670000000000096;
  Cut::Point* p3 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[2] = p3;

  x[0] = 0.94249491196456902653;
  x[1] = 0.42666670000000000984;
  x[2] = 0.29166670000000000096;
  Cut::Point* p4 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[3] = p4;

  x[0] = 0.94249721210427384044;
  x[1] = 0.42666670000000006535;
  x[2] = 0.31880103717759145088;
  Cut::Point* p5 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[4] = p5;

  x[0] = 0.94249702599927487334;
  x[1] = 0.42666670000000000984;
  x[2] = 0.32088789735568812311;
  Cut::Point* p6 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[5] = p6;

  x[0] = 0.92175085458703109875;
  x[1] = 0.42666669999999995433;
  x[2] = 0.32084499284473377845;
  Cut::Point* p7 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[6] = p7;

  x[0] = 0.91626198313675544238;
  x[1] = 0.42666670000000000984;
  x[2] = 0.32083212917404180242;
  Cut::Point* p8 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[7] = p8;

  x[0] = 0.91628195746516871711;
  x[1] = 0.42666670000000006535;
  x[2] = 0.31289655478182087922;
  Cut::Point* p9 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[8] = p9;

  x[0] = 0.91631968416588871484;
  x[1] = 0.42666669999999989882;
  x[2] = 0.29666529690944831721;
  Cut::Point* p10 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[9] = p10;

  x[0] = 0.9163322539230824848;
  x[1] = 0.42666670000000000984;
  x[2] = 0.29166670000000000096;
  Cut::Point* p11 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[10] = p11;

  x[0] = 0.88888889999999998182;
  x[1] = 0.42666670000000000984;
  x[2] = 0.29166670000000000096;
  Cut::Point* p12 = mesh.new_point(x, nullptr, s, 0.0);
  ptlist[11] = p12;

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  for (std::vector<std::vector<Cut::Point*>>::iterator i = split.begin(); i != split.end(); i++)
  {
    std::cout << "cell\n";
    std::vector<Cut::Point*> cell = *i;
    ;
    for (std::vector<Cut::Point*>::iterator j = cell.begin(); j != cell.end(); j++)
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout << coo[0] << "\t" << coo[1] << "\t" << coo[2] << "\n";
    }
  }

  /*std::vector<Cut::Point*> cell1 = split[0];
  std::vector<Cut::Point*> cell2 = split[1];
  std::vector<Cut::Point*> cell3 = split[2];
  std::vector<Cut::Point*> cell4 = split[3];
  std::vector<Cut::Point*> cell5 = split[4];
  if( cell1[0]!=p10 || cell1[1]!=p1 || cell1[2]!=p2 )
    FOUR_C_THROW( "triangulation failed for check10nodedShiftEarClipToSplit" );
  if( cell2[0]!=p10 || cell2[1]!=p2 || cell2[2]!=p3 )
    FOUR_C_THROW( "triangulation failed for check10nodedShiftEarClipToSplit" );
  if( cell3[0]!=p4 || cell3[1]!=p5 || cell3[2]!=p6 || cell3[3]!=p7 )
    FOUR_C_THROW( "triangulation failed for check10nodedShiftEarClipToSplit" );
  if( cell4[0]!=p7 || cell4[1]!=p8 || cell4[2]!=p9 )
    FOUR_C_THROW( "triangulation failed for check10nodedShiftEarClipToSplit" );
  if( cell5[0]!=p9 || cell5[1]!=p10 || cell5[2]!=p3 || cell5[3]!=p4 )
    FOUR_C_THROW( "triangulation failed for check10nodedShiftEarClipToSplit" );*/
}

void checkTemporary2(Cut::Mesh& mesh, Cut::Element* e, Cut::Side* s)
{
  std::cout << "checkTemporary2...\n";
  std::vector<Cut::Point*> ptlist;
  double x[3];
  x[2] = 0.0;

  int num = 10;
  ptlist.resize(num);

  for (int i = 0; i < num; i++)
  {
    double theta = i * 2.0 * 22.0 / 7.0 / num;
    x[0] = cos(theta);
    x[1] = sin(theta);
    Cut::Point* p1 = mesh.new_point(x, nullptr, s, 0.0);
    ptlist[i] = p1;
  }

  Cut::TriangulateFacet tf(ptlist);
  tf.split_facet();

  std::vector<std::vector<Cut::Point*>> split;
  split = tf.get_split_cells();

  for (std::vector<std::vector<Cut::Point*>>::iterator i = split.begin(); i != split.end(); i++)
  {
    std::cout << "cell\n";
    std::vector<Cut::Point*> cell = *i;
    ;
    for (std::vector<Cut::Point*>::iterator j = cell.begin(); j != cell.end(); j++)
    {
      Cut::Point* pt = *j;
      double coo[3];
      pt->coordinates(coo);
      std::cout << coo[0] << "\t" << coo[1] << "\t" << coo[2] << "\n";
    }
  }
}
