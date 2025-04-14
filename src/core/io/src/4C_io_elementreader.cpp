// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_elementreader.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_io_input_file.hpp"
#include "4C_io_value_parser.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Core::IO::ElementReader::ElementReader(std::shared_ptr<Core::FE::Discretization> dis,
    Core::IO::InputFile& input, std::string sectionname)
    : name_(dis->name()),
      input_(input),
      comm_(dis->get_comm()),
      sectionname_(sectionname),
      dis_(dis)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::ElementReader::read_and_distribute()
{
  const int myrank = Core::Communication::my_mpi_rank(comm_);
  const int numproc = Core::Communication::num_mpi_ranks(comm_);

  // read global ids of elements of this discretization
  const auto& [numele, eids] = get_element_size_and_ids();

  // determine a preliminary element distribution
  int nblock, mysize, bsize;
  {
    if (numele == 0)
    {
      // If the element section is empty, we create an empty input and return
      coleles_ = roweles_ = colnodes_ = rownodes_ = std::make_shared<Core::LinAlg::Map>(
          -1, 0, nullptr, 0, Core::Communication::as_epetra_comm(comm_));

      return;
    }

    // number of element chunks to split the reading process in
    // approximate block size (just a guess!)
    nblock = numproc;
    bsize = numele / nblock;

    // create a simple (pseudo linear) map
    mysize = bsize;
    if (myrank == numproc - 1) mysize = numele - (numproc - 1) * bsize;

    // construct the map
    roweles_ = std::make_shared<Core::LinAlg::Map>(
        -1, mysize, &eids[myrank * bsize], 0, Core::Communication::as_epetra_comm(comm_));
  }

  // define blocksizes for blocks of elements we read
  {
    // for block sizes larger than about 250000 elements (empirical value !) the code sometimes
    // hangs during ExportRowElements call for the second block (block 1). Therefore an upper limit
    // of 100000 for bsize is ensured below.
    const int maxblocksize = 100000;

    if (bsize > maxblocksize)
    {
      // without an additional increase of nblock by 1 the last block size
      // could reach a maximum value of (2*maxblocksize)-1, potentially
      // violating the intended upper limit!
      nblock = 1 + numele / maxblocksize;
      bsize = maxblocksize;
    }
  }

  get_and_distribute_elements(nblock, bsize);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::pair<int, std::vector<int>> Core::IO::ElementReader::get_element_size_and_ids() const
{
  // vector of all global element ids
  std::vector<int> eids;
  int numele = 0;

  // all reading is done on proc 0
  if (Core::Communication::my_mpi_rank(comm_) == 0)
  {
    for (const auto& element_line : input_.in_section_rank_0_only(sectionname_))
    {
      std::istringstream t{std::string{element_line.get_as_dat_style_string()}};
      int elenumber;
      std::string eletype;
      t >> elenumber >> eletype;
      elenumber -= 1;

      // only read registered element types or all elements if nothing is registered
      eids.push_back(elenumber);
    }
    numele = static_cast<int>(eids.size());
  }

  // Simply allreduce the element ids
  Core::Communication::broadcast(&numele, 1, 0, comm_);

  eids.resize(numele);
  Core::Communication::broadcast(&eids[0], numele, 0, comm_);

  return {numele, eids};
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Core::IO::ElementReader::get_and_distribute_elements(const int nblock, const int bsize)
{
  Core::Elements::ElementDefinition ed;
  ed.setup_valid_element_lines();

  // All ranks > 0 will receive the node ids of the elements from rank 0.
  // We know that we will read nblock blocks of elements, so call the
  // collective function an appropriate number of times.
  if (Core::Communication::my_mpi_rank(comm_) > 0)
  {
    for (int i = 0; i < nblock; ++i)
    {
      std::vector<int> gidlist;
      dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
    }
  }
  // Rank 0 does the actual work
  else
  {
    std::vector<int> gidlist;
    gidlist.reserve(bsize);
    int bcount = 0;
    int block = 0;

    for (const auto& element_line : input_.in_section_rank_0_only(sectionname_))
    {
      ValueParser parser{element_line.get_as_dat_style_string(),
          {.user_scope_message = "While reading element line: "}};
      const int elenumber = parser.read<int>() - 1;
      gidlist.push_back(elenumber);

      const auto eletype = parser.read<std::string>();

      // Only peek at the distype since the elements later want to parse this value themselves.
      const std::string distype = std::string(parser.peek());

      // let the factory create a matching empty element
      std::shared_ptr<Core::Elements::Element> ele =
          Core::Communication::factory(eletype, distype, elenumber, 0);
      if (!ele) FOUR_C_THROW("element creation failed");

      // For the time being we support old and new input facilities. To
      // smooth transition.

      const auto& linedef = ed.element_lines(eletype, distype);

      Core::IO::ValueParser element_parser{
          parser.get_unparsed_remainder(), {.user_scope_message = "While reading element data: "}};
      Core::IO::InputParameterContainer data;
      linedef.fully_parse(element_parser, data);

      ele->set_node_ids(distype, data);
      ele->read_element(eletype, distype, data);

      // add element to discretization
      dis_->add_element(ele);

      // get the node ids of this element
      const int numnode = ele->num_node();
      const int* nodeids = ele->node_ids();

      // all node gids of this element are inserted into a set of
      // node ids --- it will be used later during reading of nodes
      // to add the node to one or more discretisations
      std::copy(nodeids, nodeids + numnode, std::inserter(nodes_, nodes_.begin()));

      ++bcount;

      // Distribute the block if it is full. Never distribute the last block here because it
      // could be longer than expected and is therefore always distributed at the end.
      if (block != nblock - 1 && bcount == bsize)
      {
        dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
        gidlist.clear();
        bcount = 0;
        ++block;
      }
    }

    // Ensure that the last block is distributed. Since the loop might abort a lot earlier
    // than expected by the number of blocks, make sure to call the collective function
    // the appropriate number of times to match the action of the other ranks.
    for (; block < nblock; ++block)
    {
      dis_->proc_zero_distribute_elements_to_all(*roweles_, gidlist);
      gidlist.clear();
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
