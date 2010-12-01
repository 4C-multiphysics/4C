
#include "cut_volumecell.H"
#include "cut_integrationcell.H"

GEO::CUT::VolumeCell::VolumeCell( const std::set<Facet*> & facets,
                                  const std::map<std::pair<Point*, Point*>, std::set<Facet*> > & volume_lines,
                                  LinearElement * element )
  : element_( element ),
    facets_( facets )
{
/*

Das Ziel ist es einfache Volumen zu bekommen, die mit bekannten
Elementans�tzen integriert werden k�nnen.

- Die Volumenzellen in Elementkoordinaten formulieren
- Die Facets jeder Schnittebene werden zu einem Facetset zusammengefa�t.
- Die Facets der Schnittebenen werden, wenn sie nicht tri3 oder quad4 sind in
  Dreiecke zerlegt.

Offene Fragen:

- In welche Richtung sollten die Ebenen angesehen werden? Wie kann man das
  Schneiden der Schnittfl�che durch die Ebenen vermeiden?
- Was passiert mit (nahezu) achsparallelen Schnittfl�chen?
*/
}
