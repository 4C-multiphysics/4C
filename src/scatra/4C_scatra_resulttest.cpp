// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_resulttest.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_scatra_timint_meshtying_strategy_s2i.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
ScaTra::ScaTraResultTest::ScaTraResultTest(std::shared_ptr<ScaTraTimIntImpl> scatratimint)
    : Core::Utils::ResultTest("SCATRA"), scatratimint_(scatratimint)
{
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScaTra::ScaTraResultTest::test_node(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // care for the case of multiple discretizations of the same field type
  std::string dis = container.get<std::string>("DIS");
  if (dis != scatratimint_->discretization()->name()) return;

  int node = container.get<int>("NODE");
  node -= 1;

  int havenode(scatratimint_->discretization()->have_global_node(node));
  int isnodeofanybody(0);
  isnodeofanybody =
      Core::Communication::sum_all(havenode, scatratimint_->discretization()->get_comm());

  if (isnodeofanybody == 0)
  {
    FOUR_C_THROW("Node {} does not belong to discretization {}", node + 1,
        scatratimint_->discretization()->name());
  }
  else
  {
    if (scatratimint_->discretization()->have_global_node(node))
    {
      Core::Nodes::Node* actnode = scatratimint_->discretization()->g_node(node);

      // Here we are just interested in the nodes that we own (i.e. a row node)!
      if (actnode->owner() !=
          Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()))
        return;

      // extract name of quantity to be tested
      std::string quantity = container.get<std::string>("QUANTITY");

      // get result to be tested
      const double result = result_node(quantity, actnode);

      nerr += compare_values(result, "NODE", container);

      test_count++;
    }
  }
}

/*----------------------------------------------------------------------*
 | get nodal result to be tested                             fang 03/15 |
 *----------------------------------------------------------------------*/
double ScaTra::ScaTraResultTest::result_node(
    const std::string quantity,  //! name of quantity to be tested
    Core::Nodes::Node* node      //! node carrying the result to be tested
) const
{
  // initialize variable for result
  double result(0.);

  // extract row map from solution vector
  const Core::LinAlg::Map& phinpmap = scatratimint_->phinp()->get_map();

  // test result value of single scalar field
  if (quantity == "phi")
    result = scatratimint_->phinp()->local_values_as_span()[phinpmap.lid(
        scatratimint_->discretization()->dof(0, node, 0))];

  // test result value for a system of scalars
  else if (!quantity.compare(0, 3, "phi"))
  {
    // read species ID
    std::string k_string = quantity.substr(3);
    char* locator(nullptr);
    int k = strtol(k_string.c_str(), &locator, 10) - 1;

    // safety checks
    if (locator == k_string.c_str()) FOUR_C_THROW("Couldn't read species ID!");
    if (scatratimint_->discretization()->num_dof(0, node) <= k)
      FOUR_C_THROW("Species ID is larger than number of DOFs of node!");

    // extract result
    result = scatratimint_->phinp()->local_values_as_span()[phinpmap.lid(
        scatratimint_->discretization()->dof(0, node, k))];
  }

  // test domain or boundary flux
  else if (!quantity.compare(0, 12, "flux_domain_") or !quantity.compare(0, 14, "flux_boundary_"))
  {
    // read species ID
    std::string suffix;
    if (!quantity.compare(0, 12, "flux_domain_"))
      suffix = quantity.substr(12);
    else
      suffix = quantity.substr(14);
    char* locator(nullptr);
    int k = strtol(suffix.c_str(), &locator, 10) - 1;

    // safety checks
    if (locator == suffix.c_str()) FOUR_C_THROW("Couldn't read species ID!");
    if (scatratimint_->discretization()->num_dof(0, node) <= k)
      FOUR_C_THROW("Species ID is larger than number of DOFs of node!");

    // read spatial dimension
    ++locator;
    unsigned dim(-1);
    if (*locator == 'x')
      dim = 0;
    else if (*locator == 'y')
      dim = 1;
    else if (*locator == 'z')
      dim = 2;
    else
      FOUR_C_THROW("Invalid syntax!");

    // safety check
    if (*(++locator) != '\0') FOUR_C_THROW("Invalid syntax!");

    // extract result
    if (!quantity.compare(0, 12, "flux_domain_"))
      result = (*scatratimint_->flux_domain())(dim).local_values_as_span()[phinpmap.lid(
          scatratimint_->discretization()->dof(0, node, k))];
    else
      result = (*scatratimint_->flux_boundary())(dim).local_values_as_span()[phinpmap.lid(
          scatratimint_->discretization()->dof(0, node, k))];
  }

  // test result values for biofilm growth (scatra structure and scatra fluid)
  else if (quantity == "scstr_growth_displ_x")
    result =
        (*scatratimint_->str_growth())(0)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];
  else if (quantity == "scstr_growth_displ_y")
    result =
        (*scatratimint_->str_growth())(1)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];
  else if (quantity == "scstr_growth_displ_z")
    result =
        (*scatratimint_->str_growth())(2)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];
  else if (quantity == "scfld_growth_displ_x")
    result =
        (*scatratimint_->fld_growth())(0)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];
  else if (quantity == "scfld_growth_displ_y")
    result =
        (*scatratimint_->fld_growth())(1)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];
  else if (quantity == "scfld_growth_displ_z")
    result =
        (*scatratimint_->fld_growth())(2)
            .local_values_as_span()[phinpmap.lid(scatratimint_->discretization()->dof(0, node, 0))];

  // test scatra-scatra interface layer thickness
  else if (quantity == "s2ilayerthickness")
  {
    // extract scatra-scatra interface meshtying strategy class
    const std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> strategy =
        std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(scatratimint_->strategy());
    if (strategy == nullptr)
      FOUR_C_THROW("Couldn't extract scatra-scatra interface meshtying strategy class!");

    // extract state vector of discrete scatra-scatra interface layer thicknesses
    // depending on whether monolithic or semi-implicit solution approach is used
    std::shared_ptr<const Core::LinAlg::Vector<double>> s2igrowthvec(nullptr);
    switch (strategy->int_layer_growth_evaluation())
    {
      case Inpar::S2I::growth_evaluation_monolithic:
      {
        s2igrowthvec = strategy->growth_var_np();
        break;
      }

      case Inpar::S2I::growth_evaluation_semi_implicit:
      {
        s2igrowthvec = strategy->growth_var_n();
        break;
      }

      default:
      {
        FOUR_C_THROW("Can't test scatra-scatra interface layer thickness!");
        break;
      }
    }

    // safety check
    if (s2igrowthvec == nullptr)
      FOUR_C_THROW(
          "Couldn't extract state vector of discrete scatra-scatra interface layer thicknesses!");

    // extract result
    result =
        s2igrowthvec->local_values_as_span()[scatratimint_->discretization()->dof_row_map(2)->lid(
            scatratimint_->discretization()->dof(2, node, 0))];
  }

  // catch unknown quantity strings
  else
    FOUR_C_THROW("Quantity '{}' not supported in result test!", quantity);

  return result;
}  // ScaTra::ScaTraResultTest::ResultNode


/*-------------------------------------------------------------------------------------*
 | test special quantity not associated with a particular element or node   fang 03/15 |
 *-------------------------------------------------------------------------------------*/
void ScaTra::ScaTraResultTest::test_special(
    const Core::IO::InputParameterContainer& container, int& nerr, int& test_count)
{
  // make sure that quantity is tested only on specified discretization

  std::string disname = container.get<std::string>("DIS");
  if (disname == scatratimint_->discretization()->name())
  {
    // extract name of quantity to be tested
    std::string quantity = container.get<std::string>("QUANTITY");

    // get result to be tested on all processors
    const double result = result_special(quantity);

    // compare values on first processor
    if (Core::Communication::my_mpi_rank(scatratimint_->discretization()->get_comm()) == 0)
    {
      const int err = compare_values(result, "SPECIAL", container);
      nerr += err;
      test_count++;
    }
  }
}


/*----------------------------------------------------------------------*
 | get special result to be tested                           fang 03/15 |
 *----------------------------------------------------------------------*/
double ScaTra::ScaTraResultTest::result_special(
    const std::string quantity  //! name of quantity to be tested
) const
{
  // initialize variable for result
  double result(0.);

  // number of Newton-Raphson iterations in last time step
  if (quantity == "numiterlastnewton") result = (double)scatratimint_->iter_num();

  // number of outer coupling iterations in last time step
  else if (quantity == "numiterlastouter")
    result = (double)scatratimint_->iter_num_outer();

  // total values, mean values, relative L2 errors, or relative H1 errors of scalar fields in
  // subdomain or entire domain
  else if (!quantity.compare(0, 8, "totalphi") or !quantity.compare(0, 7, "meanphi") or
           !quantity.compare(0, 7, "L2error") or !quantity.compare(0, 7, "H1error"))
  {
    // examples of possible specifications:
    // 1.) quantity == "{{total,mean}phi,{L2,H1}error}", suffix == "": total value, mean value,
    // relative L2 error, or relative H1 error of scalar with ID 0 in entire domain (with ID
    // -1) 2.) quantity == "{{total,mean}phi,{L2,H1}error}a", suffix == "a": total value, mean
    // value, relative L2 error, or relative H1 error of scalar with ID a in entire domain (with
    // ID -1) 3.) quantity == "{{total,mean}phi,{L2,H1}error}_b", suffix == "_b": total value,
    // mean value, relative L2 error, or relative H1 error of scalar with ID 0 in subdomain with
    // ID b 4.) quantity == "{{total,mean}phi,{L2,H1}error}a_b", suffix == "a_b": total value,
    // mean value, relative L2 error, or relative H1 error of scalar with ID a in subdomain with
    // ID b other specifications are invalid and result in a FOUR_C_THROW
    std::string suffix;
    if (!quantity.compare(0, 8, "totalphi"))
      suffix = quantity.substr(8);
    else
      suffix = quantity.substr(7);

    const char* index(suffix.c_str());
    char* locator(nullptr);
    int species(0), domain(-1);

    if (*index == '\0')
    {
      // case 1 from list above: do nothing
    }

    else
    {
      if (*index != '_')
      {
        // cases 2 and 4 from list above
        // extract species ID
        species = strtol(index, &locator, 10) - 1;
        if (locator == index) FOUR_C_THROW("Couldn't read species ID!");

        // move index to position behind species ID
        index = locator;
      }

      if (*index == '\0')
      {
        // case 2 from list above: do nothing
      }

      else if (*index == '_')
      {
        // cases 3 and 4 from list above
        // move index to domain ID, i.e., to position behind underscore
        ++index;

        // extract domain ID
        domain = strtol(index, &locator, 10);
        if (locator == index) FOUR_C_THROW("Couldn't read domain ID!");
      }

      // catch invalid syntax
      else
        FOUR_C_THROW("Wrong syntax!");
    }

    // total or mean value
    if (!quantity.compare(0, 8, "totalphi") or !quantity.compare(0, 7, "meanphi"))
    {
      // extract map with relevant result from scalar transport time integrator
      const std::map<const int, std::vector<double>>* map(nullptr);
      if (!quantity.compare(0, 8, "totalphi"))
        map = &scatratimint_->total_scalars();
      else
        map = &scatratimint_->mean_scalars();

      // extract relevant result from map
      std::map<const int, std::vector<double>>::const_iterator iterator = map->find(domain);
      if (iterator == map->end() or species < 0 or species >= (int)iterator->second.size())
        FOUR_C_THROW(
            "Couldn't extract total or mean value of transported scalar with ID {} inside domain "
            "with ID {} from time integrator!",
            species, domain);
      result = iterator->second[species];
    }

    // relative L2 or H1 error
    else
    {
      // global error
      if (domain == -1)
      {
        if (!quantity.compare(0, 7, "L2error"))
          result = (*scatratimint_->rel_errors())[species * 2];
        else
          result = (*scatratimint_->rel_errors())[species * 2 + 1];
      }

      // error inside subdomain
      else
      {
        if (!quantity.compare(0, 7, "L2error"))
          result = (*scatratimint_
                  ->rel_errors())[domain * scatratimint_->num_dof_per_node() * 2 + species * 2];
        else
          result = (*scatratimint_
                  ->rel_errors())[domain * scatratimint_->num_dof_per_node() * 2 + species * 2 + 1];
      }
    }
  }

  // number of iterations performed by linear solver during last Newton-Raphson iteration
  else if (quantity == "numiterlastsolve")
  {
    result = (double)scatratimint_->strategy()->solver().get_num_iters();
  }

  // test parallel distribution of scatra-scatra coupling interface
  else if (!quantity.compare(0, 10, "s2inumdof_"))
  {
    // extract string part behind "s2inumdof_"
    // it has the structure "intx_procy". We are searching for the values of x and y
    const std::string suffix = quantity.substr(10);

    // check syntax
    if (suffix.compare(0, 3, "int") and suffix.find("_proc") == std::string::npos)
      FOUR_C_THROW("Wrong syntax!");

    const std::string interface_string = suffix.substr(3, suffix.find('_') - 3);
    const int interface_num = std::stoi(interface_string);

    std::string proc_string = suffix.substr(suffix.find("_proc") + 5);
    const int proc_num = std::stoi(proc_string);

    // extract processor ID
    if (proc_num >= Core::Communication::num_mpi_ranks(scatratimint_->discretization()->get_comm()))
      FOUR_C_THROW("Invalid processor ID!");

    // extract scatra-scatra interface meshtying strategy class
    const std::shared_ptr<const ScaTra::MeshtyingStrategyS2I> strategy =
        std::dynamic_pointer_cast<const ScaTra::MeshtyingStrategyS2I>(scatratimint_->strategy());
    if (strategy == nullptr)
      FOUR_C_THROW("Couldn't extract scatra-scatra interface meshtying strategy class!");

    // extract number of degrees of freedom owned by specified processor at specified
    // scatra-scatra coupling interface
    result = strategy->mortar_discretization(interface_num).dof_row_map()->num_my_elements();
    Core::Communication::broadcast(
        &result, 1, proc_num, scatratimint_->discretization()->get_comm());
  }

  // test relaxation parameters for partitioned simulations
  else if (!quantity.compare(0, 5, "omega"))
  {
    // read parameter index
    std::string index_string = quantity.substr(5);
    char* locator(nullptr);
    const unsigned index = strtol(index_string.c_str(), &locator, 10) - 1;

    // safety checks
    if (locator == index_string.c_str()) FOUR_C_THROW("Couldn't read parameter index!");
    if (index >= scatratimint_->omega().size()) FOUR_C_THROW("Invalid parameter index!");

    // extract result
    result = scatratimint_->omega()[index];
  }

  // test total number of time steps
  else if (!quantity.compare(0, 7, "numstep"))
    result = scatratimint_->step();

  // test domainintegral_ID
  else if (!quantity.compare(0, 15, "domainintegral_"))
  {
    std::string suffix = quantity.substr(15);
    const char* index(suffix.c_str());
    char* locator(nullptr);
    // extract domain ID
    int domain = strtol(index, &locator, 10);
    if (domain < 0 || domain > (int)(scatratimint_->domain_integrals().size() - 1))
      FOUR_C_THROW("Value for domain integral has to lie between 0 and {}",
          (int)(scatratimint_->domain_integrals().size() - 1));
    result = scatratimint_->domain_integrals()[domain];
  }

  // test boundaryintegral_ID
  else if (!quantity.compare(0, 17, "boundaryintegral_"))
  {
    std::string suffix = quantity.substr(17);
    const char* index(suffix.c_str());
    char* locator(nullptr);
    // extract boundary ID
    int boundary = strtol(index, &locator, 10);
    if (boundary < 0 || boundary > (int)(scatratimint_->boundary_integrals().size() - 1))
      FOUR_C_THROW("Value for boundary integral has to lie between 0 and {}",
          (int)(scatratimint_->domain_integrals().size() - 1));
    result = scatratimint_->boundary_integrals()[boundary];
  }

  // catch unknown quantity strings
  else
    FOUR_C_THROW("Quantity '{}' not supported in result test!", quantity);

  return result;
}  // ScaTra::ScaTraResultTest::result_special

FOUR_C_NAMESPACE_CLOSE
