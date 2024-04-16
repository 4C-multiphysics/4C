/*---------------------------------------------------------------------*/
/*! \file

\brief colored graph to create volumecells from facets

\level 3


*----------------------------------------------------------------------*/

#include "baci_cut_coloredgraph.hpp"

#include "baci_cut_facet.hpp"
#include "baci_cut_output.hpp"

#include <algorithm>
#include <fstream>
#include <queue>
#include <stack>
#include <stdexcept>

BACI_NAMESPACE_OPEN

bool CORE::GEO::CUT::COLOREDGRAPH::ForkFinder::operator()(
    const std::pair<const int, plain_int_set>& point)
{
  if (point.first < graph_.Split()) return false;

  plain_int_set& row = graph_[point.first];
  if (row.size() > 2)
  {
    plain_int_set& used_row = used_[point.first];
    for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
    {
      int p = *i;
      if (used_row.count(p) == 0)
      {
        if (free_.count(p) > 0) return true;
        if (cycle_.count(p) > 0) return true;
      }
    }
    return false;
  }
  return false;
}

// add connection in the graph
void CORE::GEO::CUT::COLOREDGRAPH::Graph::Add(int row, int col)
{
  if (row >= color_split_ and col >= color_split_) dserror("two lines connected");
  if (row < color_split_ and col < color_split_) dserror("two facets connected");
  graph_[row].insert(col);
  graph_[col].insert(row);
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::Add(int p, const plain_int_set& row)
{
  for (plain_int_set::const_iterator i = row.begin(); i != row.end(); ++i)
  {
    Add(p, *i);
  }
}

int CORE::GEO::CUT::COLOREDGRAPH::Graph::FindNext(
    Graph& used, int point, Graph& cycle, const plain_int_set& free)
{
  // find current connections of the point
  plain_int_set& row = graph_[point];
  // find which directions were already visisited
  plain_int_set& used_row = used[point];
  for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
  {
    int p = *i;
    // if not visited yet
    if (used_row.count(p) == 0)
    {
      // if it is in array of free edges
      if (free.count(p) > 0) return p;
      // if it is in array of free cycles
      if (cycle.count(p) > 0) return p;
    }
  }
  return -1;
}

// get all number of the graph in the plain int set
void CORE::GEO::CUT::COLOREDGRAPH::Graph::GetAll(plain_int_set& all)
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int p = i->first;
    all.insert(p);
  }
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::FixSingleLines()
{
  for (;;)
  {
    std::map<int, plain_int_set>::iterator j =
        std::find_if(graph_.begin(), graph_.end(), SingeLineFinder(color_split_));
    if (j == graph_.end())
    {
      return;
    }

    int p1 = j->first;
    plain_int_set& row = j->second;
    for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
    {
      int p2 = *i;
      plain_int_set& row2 = graph_[p2];
      row2.erase(p1);
      if (row2.size() == 0) graph_.erase(p2);
    }
    graph_.erase(j);
  }
}



// Test if all edges of the graph has more than 1 connection (then it is closed)
void CORE::GEO::CUT::COLOREDGRAPH::Graph::TestClosed()
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    plain_int_set& row = i->second;
    if (row.size() < 2)
    {
      // -------------------------------------------------------------------    (sudhakar 08/14)
      // one of the possible reasons this may occur is the following:
      // A background element is cut by two cut sides, that are not connected
      // This is not a multiple cut situation ==> this works with cut algorithm
      // See the 2D example here
      //
      //        ++++++++++++++++              ++++++++++++++
      //        +              +              +            +
      //        +              +              +            +
      //  o------------------------o          +            +
      //        +              +          o-------o   o-----------o
      //        +              +              +            +
      //  o------------------------o          +            +
      //        +              +              +            +
      //        ++++++++++++++++              ++++++++++++++
      //     (multiple cut --> okay)     (open-point in colored graph)
      //
      // In such situations, geometrically two separate volumecells can't be formed
      // Check your cut_mesh from cut_mesh*.pos
      // -------------------------------------------------------------------
      dserror("open point in colored graph ( facet-id = %d )", i->first);
      dserror("open point in colored graph");
    }
  }
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::TestFacets()
{
  for (std::map<int, plain_int_set>::iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int p = i->first;
    // for all facets
    if (p < color_split_)
    {
      plain_int_set& row = i->second;
      if (row.size() < 3)
      {
        dserror("facets need at least three lines");
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CORE::GEO::CUT::COLOREDGRAPH::Graph::Print() const
{
  for (std::map<int, plain_int_set>::const_iterator i = graph_.begin(); i != graph_.end(); ++i)
  {
    int p = i->first;
    const plain_int_set& row = i->second;
    std::cout << p << ": ";
    for (plain_int_set::const_iterator j = row.begin(); j != row.end(); ++j)
    {
      int pp = *j;
      std::cout << pp << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

namespace CORE::GEO
{
  namespace CUT
  {
    namespace COLOREDGRAPH
    {
      bool IsFree(Graph& used, plain_int_set& free, int i)
      {
        return used.count(i) == 0 and free.count(i) > 0;
      }

      int FindFirstFreeFacet(Graph& graph, Graph& used, plain_int_set& free)
      {
        for (plain_int_set::iterator i = free.begin(); i != free.end(); ++i)
        {
          int facet = *i;
          if (facet >= graph.Split())
          {
            dserror("no free facet but free lines");
          }
          plain_int_set& row = graph[facet];
          // check if this facet visited connected lines to it
          for (const int& line : row)
          {
            // if this facet is free, but the line connected to it is not free,
            // this means that lines connected to it leads outside, hence it is "first" free facet
            if (not IsFree(used, free, line))
            {
              return facet;
            }
          }
        }
        dserror("empty free set");
      }

      bool IsValidFacet(plain_int_set& row, const std::vector<int>& visited)
      {
        for (const int& line : row)
        {
          // if it is visited more than once
          if (visited[line] >= 2)
          {
            // dserror("Invalid facet!");
            dserror("Invalid facet");
            return false;
          }
        }
        return true;
      }

      void MarkFacet(Graph& used, plain_int_set& free, int facet, plain_int_set& row,
          std::vector<int>& visited, int& num_split_lines)
      {
        // mark facet and lines
        if (facet >= used.Split())  // means this is a line
        {
          dserror("This should not happen");
        }
        visited[facet] += 1;
        if (visited[facet] > 1) dserror("facet visited more than once");
        // iterate over lines connected to this facet
        for (const int& line : row)
        {
          if (IsFree(used, free, line))
          {
            if (visited[line] == 0) num_split_lines += 1;
            // was connected by another facet already
            else if (visited[line] == 1)
              num_split_lines -= 1;
          }
          // visit it
          visited[line] += 1;
          // was visited by more than two facets
          if (visited[line] > 2) dserror("too many facets at line");
        }
      }

      void UnMarkFacet(Graph& used, plain_int_set& free, int facet, plain_int_set& row,
          std::vector<int>& visited, int& num_split_lines)
      {
        // unmark facet and lines
        for (plain_int_set::iterator i = row.begin(); i != row.end(); ++i)
        {
          int line = *i;
          if (IsFree(used, free, line))
          {
            if (visited[line] == 1)
              num_split_lines -= 1;
            else if (visited[line] == 2)
              num_split_lines += 1;
          }
          visited[line] -= 1;
          if (visited[line] < 0) dserror("too few facets at line");
        }
        visited[facet] -= 1;
        if (visited[facet] < 0) dserror("facet left more than once");
      }

      bool VisitFacetDFS(Graph& graph, Graph& used, plain_int_set& free, int facet,
          const std::vector<std::pair<Point*, Point*>>& all_lines, std::vector<int>& visited,
          std::vector<int>& split_trace)
      {
        try
        {
          std::vector<int> facet_stack;
          facet_stack.push_back(facet);

          int num_split_lines = 0;
          std::vector<int> facet_color(graph.size(), 0);

          // performing depth-first graph traversal while there is not more facets left
          while (not facet_stack.empty())
          {
            facet = facet_stack.back();
            facet_stack.pop_back();

            if (facet_color[facet] == 0)  // white - not visited before
            {
              plain_int_set& row = graph[facet];

              if (IsValidFacet(row, visited))
              {
                MarkFacet(used, free, facet, row, visited, num_split_lines);

                facet_color[facet] = 1;
                facet_stack.push_back(facet);

                // test for success: only non-free lines are open, meaning, all the internal lines
                // have been visited and found matching facets
                if (num_split_lines == 0)
                {
                  // build split_trace

                  // iterate over all visited lines
                  for (std::vector<int>::iterator i = visited.begin() + graph.Split();
                       i != visited.end(); ++i)
                  {
                    // if it was visited only once, meaning no other facet visited this line, means
                    // it is open
                    if (*i == 1)
                    {
                      // getting its id
                      int line = i - visited.begin();
                      // if lies not inside
                      if (not IsFree(used, free, line))
                      {
                        split_trace.push_back(line);
                      }
                    }
                  }
                  // otherwise throw error, we could not find what lines split the volume
                  if (split_trace.size() == 0)
                  {
                    dserror("no split trace");
                  }

                  return true;
                }

                // try neighbouring facets
                for (const int& line : row)
                {
                  // facet not visited and does not lead us outside
                  if (visited[line] < 2 and IsFree(used, free, line))
                  {
                    plain_int_set& row = graph[line];
                    // get all the facets connected to that line and push it for traversal
                    for (const int& f : row)
                    {
                      if (facet_color[f] == 0 and IsFree(used, free, f))
                      {
                        facet_stack.push_back(f);
                      }
                    }
                  }
                }
              }
            }
            else if (facet_color[facet] == 1)  // gray - we finished all the the "tree" of this
                                               // facet - need to travserse something else
            {
              // search for any facet within the stack that can be added
              int pos = facet_stack.size() - 1;
              for (; pos >= 0; --pos)
              {
                int f = facet_stack[pos];
                if (facet_color[f] == 0 and IsValidFacet(graph[f], visited))
                {
                  facet_stack.push_back(facet);
                  facet_stack.push_back(f);
                  break;
                }
              }

              // clear current facet if there is nothing more to add
              if (pos < 0)
              {
                plain_int_set& row = graph[facet];
                UnMarkFacet(used, free, facet, row, visited, num_split_lines);

                if (std::find(facet_stack.begin(), facet_stack.end(), facet) == facet_stack.end())
                  facet_color[facet] = 0;
                else
                  facet_color[facet] = 2;
              }
            }
            else if (facet_color[facet] == 2)  // black
            {
              // if a black facet is poped for the last time (it is not any more
              // on the stack), we make it available again.
              if (std::find(facet_stack.begin(), facet_stack.end(), facet) == facet_stack.end())
              {
                facet_color[facet] = 0;
              }
            }
          }
        }
        catch (CORE::Exception& err)
        {
          std::cout << "Failed in the colored graph in the DFS search" << std::endl;
          std::cout << "Last processed facet is" << facet << std::endl;
          std::ofstream file("facetgraph_failed.pos");
          for (int facet_id = 0; facet_id < facet; ++facet_id)
          {
            std::stringstream section_name;
            section_name << "Facet" << facet_id;
            CORE::GEO::CUT::OUTPUT::GmshNewSection(file, section_name.str());
            CORE::GEO::CUT::OUTPUT::GmshFacetDump(file,
                static_cast<CORE::GEO::CUT::Facet*>(graph.GetPointer(facet_id)), "lines", true,
                false, nullptr);
            CORE::GEO::CUT::OUTPUT::GmshEndSection(file, false);
          }
          file.close();

          std::ofstream filenext("facetgraph_failed_last_facet.pos");
          CORE::GEO::CUT::OUTPUT::GmshNewSection(filenext, "Facets");
          CORE::GEO::CUT::OUTPUT::GmshFacetDump(filenext,
              static_cast<CORE::GEO::CUT::Facet*>(graph.GetPointer(facet)), "lines", true, false,
              nullptr);
          CORE::GEO::CUT::OUTPUT::GmshEndSection(filenext, true);


          std::cout << "Point IDs of failed facet are " << std::endl;
          static_cast<CORE::GEO::CUT::Facet*>(graph.GetPointer(facet))->PrintPointIds();

          std::ofstream filevisited("facetgraph_visited_facets.pos");
          for (std::vector<int>::iterator i = visited.begin(); i != visited.begin() + graph.Split();
               ++i)
          {
            if (*i == 1)
            {
              int facet = i - visited.begin();
              CORE::GEO::CUT::OUTPUT::GmshNewSection(filevisited, "Facets");
              CORE::GEO::CUT::OUTPUT::GmshFacetDump(filevisited,
                  static_cast<CORE::GEO::CUT::Facet*>(graph.GetPointer(facet)), "lines", true,
                  false, nullptr);
              CORE::GEO::CUT::OUTPUT::GmshEndSection(filevisited, false);
            }
          }
          filevisited.close();
          dserror("");
        }
        return false;
      }

    }  // namespace COLOREDGRAPH
  }    // namespace CUT
}  // namespace CORE::GEO

void CORE::GEO::CUT::COLOREDGRAPH::Graph::FindFreeFacets(Graph& graph, Graph& used,
    plain_int_set& free, const std::vector<std::pair<Point*, Point*>>& all_lines,
    std::vector<int>& split_trace)
{
  int free_facet = FindFirstFreeFacet(graph, used, free);

  std::vector<int> visited(graph.size(), 0);

  if (not VisitFacetDFS(graph, used, free, free_facet, all_lines, visited, split_trace))
  {
    dserror("Failed to find volume split. DFS search failed");
  }

  // iterate over all facets that are visited and check ( only internal should be visited )
  for (std::vector<int>::iterator i = visited.begin(); i != visited.begin() + graph.Split(); ++i)
  {
    if (*i == 1)
    {
      int facet = i - visited.begin();
      plain_int_set& row = graph[facet];
      for (const int& line : row)
      {
        used.Add(line, facet);
        Add(line, facet);
        free.erase(line);
      }
      // erased from internal free facets
      free.erase(facet);
    }
    else if (*i > 1)
    {
      dserror("same facet was visited twice");
    }
  }
}


// find set of lines which are connected only to one facet
void CORE::GEO::CUT::COLOREDGRAPH::Graph::FindSplitTrace(std::vector<int>& split_trace)
{
  for (Graph::const_iterator i = begin(); i != end(); ++i)
  {
    int p = i->first;
    const plain_int_set& row = i->second;
    if (p >= Split())
    {
      if (row.size() == 1)
      {
        split_trace.push_back(p);
      }
    }
  }
}

// check if the graph contain given set of edges
bool CORE::GEO::CUT::COLOREDGRAPH::Graph::ContainsTrace(const std::vector<int>& split_trace)
{
  for (std::vector<int>::const_iterator i = split_trace.begin(); i != split_trace.end(); ++i)
  {
    int p = *i;
    if (count(p) == 0) return false;
  }
  return true;
}

void CORE::GEO::CUT::COLOREDGRAPH::Cycle::Print() const
{
  std::cout << "Cycle:\n";
  cycle_.Print();
  std::cout << "\n";
}

// splits splittrace into isolated components, and pushes it into splitted_trace
void CORE::GEO::CUT::COLOREDGRAPH::Graph::SplitSplittrace(const std::vector<int>& split_trace,
    Graph& datagraph, std::vector<std::vector<int>>& isolated_components)
{
  // first construct point -> line relations from the datagraph
  std::map<Point*, std::vector<int>> point_line_map;
  for (std::vector<int>::const_iterator it = split_trace.begin(); it != split_trace.end(); ++it)
  {
    int line_id = *it;
    if (line_id < Split())
    {
      dserror("Only lines are allowed in the split trace!");
    }
    else
    {
      // get our lines to the map
      std::pair<Point*, Point*> line =
          *static_cast<std::pair<Point*, Point*>*>(datagraph.GetPointer(line_id));
      point_line_map[line.first].push_back(line_id);
      point_line_map[line.second].push_back(line_id);
    }
  }

  // second pass, now constructor connectivity

  std::set<int> split_trace_set(split_trace.begin(), split_trace.end());

  std::stack<int> lines_to_explore;
  unsigned int seed = split_trace.front();
  lines_to_explore.push(seed);
  bool visited_all_loops = false;

  // lines that we proccessed over (not to
  // go though them again if we have connected split loops)

  std::set<int> done_lines;
  // while there are still isolated loops in the split trace
  while (not visited_all_loops)
  {
    std::set<int> visited;

    while (not lines_to_explore.empty())
    {
      unsigned int l = lines_to_explore.top();
      lines_to_explore.pop();
      // get our line
      std::pair<Point*, Point*> line =
          *static_cast<std::pair<Point*, Point*>*>(datagraph.GetPointer(l));
      // get lines connected to it
      const std::vector<int>& end_lines = point_line_map[line.second];
      const std::vector<int>& front_lines = point_line_map[line.first];
      std::vector<int> connected_lines(front_lines);
      connected_lines.insert(connected_lines.begin(), end_lines.begin(), end_lines.end());
      // remove connected to 'l' line itself
      connected_lines.erase(
          std::remove(connected_lines.begin(), connected_lines.end(), l), connected_lines.end());
      if (connected_lines.size() != 2)
      {
        // check if most of lines come from the same point ( of course there is one more line
        // connected on the other end)
        if (end_lines.size() == 2 or front_lines.size() == 2)
        {
          connected_lines = (end_lines.size() < front_lines.size() ? end_lines : front_lines);
          for (std::vector<int>::iterator it = connected_lines.begin(); it != connected_lines.end();
              /**/)
          {
            if (done_lines.find(*it) != done_lines.end())
            {
              it = connected_lines.erase(it);
            }
            else
              ++it;
          }
        }
        else
        {
          dserror(
              "Unknown case of the line in split trace connected to %d lines at the same time"
              "It should be 2",
              connected_lines.size());
        }
      }

      visited.insert(l);
      for (const int& connected_line : connected_lines)
      {
        if (visited.count(connected_line) == 0)
        {
          lines_to_explore.push(connected_line);
        }
      }
    }

    std::set<int> split_trace_set_new;

    std::set_difference(split_trace_set.begin(), split_trace_set.end(), visited.begin(),
        visited.end(), std::inserter(split_trace_set_new, split_trace_set_new.end()));

    // need to push_back difference to the list
    isolated_components.push_back(std::vector<int>(visited.begin(), visited.end()));
    done_lines.insert(visited.begin(), visited.end());

    std::swap(split_trace_set_new, split_trace_set);

    if (split_trace_set.empty())
    {
      visited_all_loops = true;
    }

    else
    {
      lines_to_explore.push(*split_trace_set.begin());
    }
  }
#if EXTENDED_CUT_DEBUG_OUTPUT
  std::cout << "Number of isolated loops is " << isolated_components.size() << std::endl;
#endif
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::Split(Graph& used, plain_int_set& free, Graph& connection,
    const std::vector<int>& split_trace, Graph& c1, Graph& c2, Graph& datagraph)
{
  // find lhs and rhs starting from split trace

  plain_int_set* facet_row = nullptr;
  plain_int_set* facet_tmp_row = nullptr;
  for (const int& line : split_trace)
  {
    facet_tmp_row = &at(line);
    if (facet_tmp_row->size() == 2)
    {
      facet_row = facet_tmp_row;
    }
  }
  if (facet_row == nullptr) facet_row = facet_tmp_row;  // last passed

  if (facet_row->size() != 2)
  {
    // This might happen and it might be a valid split. How to deal with it?
    dserror("expect two facets at line");
  }

  plain_int_set::iterator facet_it = facet_row->begin();

  Fill(split_trace, used, free, connection, *facet_it, c1);
  ++facet_it;
  Fill(split_trace, used, free, connection, *facet_it, c2);


  // detect anomalies, where split line is connected to facets from both cycles
  auto not_connected_line = std::find_if(split_trace.begin(), split_trace.end(),
      [&c1, &c2](int line) { return (c1[line].size() == 1 or c2[line].size() == 1); });
  // we are fine
  if (not_connected_line == split_trace.end()) return;

  // try to detect which cycles lacks closing facets and add them
  else
  {
    // it might happen, when we have holes in the split trace , we try to generate corresponding
    // cycles as well
    std::vector<std::vector<int>> isolated_components;
    SplitSplittrace(split_trace, datagraph, isolated_components);

    unsigned int n_components = isolated_components.size();
    if (n_components == 1)
      dserror(
          "Number of isolated components of the split trace is, but graph is open anyway. Check "
          "this case");
    else if (n_components > 2)
    {
      std::cout << "WARNING: Number of isolated components of the split trace is " << n_components
                << "this case probably will work fine, but it was not detaily though of. In case "
                   "of problem, check output"
                << std::endl;
    }

    for (const int& line : split_trace)
    {
      unsigned c1_connected_facets = c1[line].size();
      unsigned c2_connected_facets = c2[line].size();

      Graph* fine_cycle = nullptr;
      Graph* open_cycle = nullptr;

      if (c1_connected_facets > 1 and c2_connected_facets > 1)
      {
        // fine
      }
      else if (c1_connected_facets == 1 and c2_connected_facets > 1)
      {
        fine_cycle = &c2;
        open_cycle = &c1;
      }
      else if (c1_connected_facets > 1 and c2_connected_facets == 1)
      {
        fine_cycle = &c1;
        open_cycle = &c2;
      }
      else
      {
        dserror("open line after graph split");
      }

      if (open_cycle != nullptr)
      {
#if EXTENDED_CUT_DEBUG_OUTPUT
        std::cout << "NOTICE: One of the graph split results is open" << std::endl;
#endif
        // Find split trace component where this line belongs too
        // If speed up is needed, we can just first filter the not-connected lines and then
        // match them with the isolated_components of the split trace
        std::vector<std::vector<int>>::const_iterator c_it = isolated_components.begin();
        for (; c_it != isolated_components.end(); ++c_it)
        {
          const std::vector<int>& component = *c_it;
          // try to find line on this component
          if (std::find(component.begin(), component.end(), line) != component.end()) break;
        }

        if (c_it == isolated_components.end())
          dserror(
              "Could not find isolated component of the split trace where line belongs to. Check "
              "this case");

        const std::vector<int>& isolated_split_trace = *c_it;

        plain_int_set& facet_row = at(line);
        if (facet_row.size() != 2) dserror("Expect two facets at line");
        plain_int_set::iterator i = facet_row.begin();
        int f1 = *i;
        ++i;
        int f2 = *i;
        // select where to start the split
        if (fine_cycle->count(f1) > 0)
        {
          Fill(isolated_split_trace, used, free, connection, f2, *open_cycle);
        }
        else if (fine_cycle->count(f2) > 0)
        {
          Fill(isolated_split_trace, used, free, connection, f1, *open_cycle);
        }
        else
        {
          dserror("Could fill found the seed facet for cycle creation");
        }
      }
    }
  }
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::Fill(const std::vector<int>& split_trace, Graph& used,
    plain_int_set& free, Graph& connection, int seed, Graph& c)
{
  plain_int_set visited;
  visited.insert(split_trace.begin(), split_trace.end());
  std::stack<int> facets_to_explore;

  facets_to_explore.push(seed);

  while (not facets_to_explore.empty())
  {
    int f = facets_to_explore.top();
    facets_to_explore.pop();

    visited.insert(f);

    plain_int_set& lines_row = at(f);
    for (const int& line : lines_row)
    {
      c.Add(f, line);
      // if we have not come to split trace again ( finished )
      if (visited.count(line) == 0)  // and IsFree( used, free, line ) )
      {
        plain_int_set& facets_row = at(line);
        // discover new facet and visit it
        for (const int& f : facets_row)
        {
          if (visited.count(f) == 0)
          {
            facets_to_explore.push(f);
          }
        }
      }
    }
  }

  // adding internal connection to close the cycle
  for (Graph::const_iterator i = connection.begin(); i != connection.end(); ++i)
  {
    int line = i->first;
    const plain_int_set& facets = i->second;
    c.Add(line, facets);
  }
}

void CORE::GEO::CUT::COLOREDGRAPH::CycleList::AddPoints(Graph& graph, Graph& used, Graph& cycle,
    plain_int_set& free, const std::vector<std::pair<Point*, Point*>>& all_lines)
{
  PushBack(cycle);

  // while there are elements to loop over  ( while there are still "cutted-parts" of the element)
  while (free.size() > 0)
  {
    // create new graph with the same separation of line and facets
    Graph connection(graph.Split());

    // find connection graph and trace lines
    std::vector<int> split_trace;
    connection.FindFreeFacets(graph, used, free, all_lines, split_trace);

    // There might be multiple matches. Only one of those is the one we are
    // looking for.
    std::vector<std::list<Cycle>::iterator> matching;
    for (std::list<Cycle>::iterator i = cycles_.begin(); i != cycles_.end(); ++i)
    {
      Cycle& c = *i;
      if (c.ContainsTrace(split_trace))
      {
        matching.push_back(i);
      }
    }

    bool found = false;

    for (std::vector<std::list<Cycle>::iterator>::iterator ilist = matching.begin();
         ilist != matching.end(); ++ilist)
    {
      Cycle& c = **ilist;

      Graph c1(graph.Split());
      Graph c2(graph.Split());

      // split the cycle into two cycles based on 'split trace'
      c.Split(used, free, connection, split_trace, c1, c2, graph);

      if (c1 == c2)
      {
        if (matching.size() == 1)
        {
          PushBack(c1);

          // this is happens when after finding split trace, both division along it produces same
          // cycles
          dserror("bad luck");

          cycles_.erase(*ilist);
          found = true;
          break;
        }
      }
      else
      {
        // sanity test
        for (Graph::const_iterator i = c().begin(); i != c().end(); ++i)
        {
          int f = i->first;
          if (f >= c().Split()) break;
          if (connection.count(f) == 0 and c1.count(f) > 0 and c2.count(f) > 0)
          {
            dserror("not a valid split");
          }
        }

        PushBack(c1);
        PushBack(c2);

        cycles_.erase(*ilist);
        found = true;
        break;
      }
    }
    if (not found)
    {
      dserror("did not find volume that contains split facets");
    }
  }
}

void CORE::GEO::CUT::COLOREDGRAPH::CycleList::PushBack(Graph& g)
{
  cycles_.push_back(Cycle(g.Split()));
  Cycle& c = cycles_.back();
  c.Assign(g);
}

void CORE::GEO::CUT::COLOREDGRAPH::CycleList::Print() const
{
  for (const Cycle& c : cycles_) c.Print();
}

void CORE::GEO::CUT::COLOREDGRAPH::Graph::DumpGraph(const std::string& name)
{
  std::ofstream file(name.c_str());
  file << "color_split = " << Split() << "\n";
  file << "graph = [";
  for (const_iterator i = begin(); i != end(); ++i)
  {
    file << i->first << ",";
  }
  file << "]\n";
  file << "data = {\n";
  for (const_iterator i = begin(); i != end(); ++i)
  {
    file << "    " << i->first << ": [";
    std::copy(i->second.begin(), i->second.end(), std::ostream_iterator<int>(file, ","));
    file << "],\n";
  }
  file << "}\n";
}

BACI_NAMESPACE_CLOSE
