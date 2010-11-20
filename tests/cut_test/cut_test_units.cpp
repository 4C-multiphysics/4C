
#include "../drt_cut/cut_mesh.H"
#include "../drt_cut/cut_intersection.H"
#include "cut_test_utils.H"

void test_unit_intersection_touch()
{
  double scale = 1.1e-6;
  for ( int i=0; i<7; ++i )
  {
    double x = pow( 0.1, i );
    GEO::CUT::Mesh mesh( x );

    Epetra_SerialDenseMatrix xyze( 3, 4 );

    xyze( 0, 0 ) = 0;
    xyze( 1, 0 ) = 0;
    xyze( 2, 0 ) = 0;

    xyze( 0, 1 ) = x;
    xyze( 1, 1 ) = 0;
    xyze( 2, 1 ) = 0;

    xyze( 0, 2 ) = x;
    xyze( 1, 2 ) = 0;
    xyze( 2, 2 ) = x;

    xyze( 0, 3 ) = 0;
    xyze( 1, 3 ) = 0;
    xyze( 2, 3 ) = x;

    GEO::CUT::Side * s1 = create_quad4( mesh, xyze );

    xyze( 0, 0 ) = 0;
    xyze( 1, 0 ) = -scale*x;
    xyze( 2, 0 ) = 0;

    xyze( 0, 1 ) = 0;
    xyze( 1, 1 ) = x;
    xyze( 2, 1 ) = 0;

    xyze( 0, 2 ) = 0;
    xyze( 1, 2 ) = x;
    xyze( 2, 2 ) = x;

    xyze( 0, 3 ) = 0;
    xyze( 1, 3 ) = scale*x;
    xyze( 2, 3 ) = x;

    GEO::CUT::Side * s2 = create_quad4( mesh, xyze );

    const std::vector<GEO::CUT::Edge*> & edges = s2->Edges();

    GEO::CUT::Edge* e = edges[3];

    if ( e->Nodes()[0]->point()->Id()!=7 or e->Nodes()[1]->point()->Id()!=4 )
    {
      throw std::runtime_error( "unexpected nodal id" );
    }

    GEO::CUT::Intersection<DRT::Element::line2, DRT::Element::quad4>
      intersection( mesh,
                    *dynamic_cast<GEO::CUT::ConcreteEdge<DRT::Element::line2>*>( e ),
                    *dynamic_cast<GEO::CUT::ConcreteSide<DRT::Element::quad4>*>( s1 ) );

    std::set<GEO::CUT::Point*, GEO::CUT::PointPidLess> cuts;
    intersection.Intersect( cuts );

    for ( std::set<GEO::CUT::Point*, GEO::CUT::PointPidLess>::iterator i=cuts.begin();
          i!=cuts.end();
          ++i )
    {
      GEO::CUT::Point* p = *i;
      if ( p->Id()!=8 )
      {
        throw std::runtime_error( "unexpected nodal id" );
      }
    }
  }
}
