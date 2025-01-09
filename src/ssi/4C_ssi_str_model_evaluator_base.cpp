// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_ssi_str_model_evaluator_base.hpp"

#include "4C_adapter_str_ssiwrapper.hpp"
#include "4C_comm_exporter.hpp"
#include "4C_fem_general_utils_gauss_point_postprocess.hpp"
#include "4C_io.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"


FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BaseSSI::determine_stress_strain()
{
  // extract raw data for element-wise stresses
  const std::vector<char>& stressdata = eval_data().stress_data();

  // initialize map for element-wise stresses
  const auto stresses =
      std::make_shared<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>();

  Core::Communication::UnpackBuffer buffer(stressdata);
  // loop over all row elements
  for (int i = 0; i < discret().element_row_map()->NumMyElements(); ++i)
  {
    // initialize matrix for stresses associated with current element
    const auto stresses_ele = std::make_shared<Core::LinAlg::SerialDenseMatrix>();

    // extract stresses
    extract_from_pack(buffer, *stresses_ele);

    // store stresses
    (*stresses)[discret().element_row_map()->GID(i)] = stresses_ele;
  }

  // export map to column format
  Core::Communication::Exporter exporter(
      *discret().element_row_map(), *discret().element_col_map(), discret().get_comm());
  exporter.do_export(*stresses);

  // prepare nodal stress vectors
  Core::LinAlg::MultiVector<double> nodal_stresses_source(*discret().node_row_map(), 6);

  discret().evaluate(
      [&](Core::Elements::Element& ele)
      {
        Core::FE::extrapolate_gauss_point_quantity_to_nodes(
            ele, *stresses->at(ele.id()), discret(), nodal_stresses_source);
      });

  const auto* nodegids = discret().node_row_map();
  for (int i = 0; i < nodegids->NumMyElements(); ++i)
  {
    const int nodegid = nodegids->GID(i);

    // extract lid of node as multi-vector is sorted according to the node ids
    const Core::Nodes::Node* const node = discret().g_node(nodegid);
    const int nodelid = discret().node_row_map()->LID(nodegid);

    // extract dof lid of first degree of freedom associated with current node in second nodeset
    const int dofgid_epetra = discret().dof(2, node, 0);
    const int doflid_epetra = mechanical_stress_state_np_->get_block_map().LID(dofgid_epetra);
    if (doflid_epetra < 0) FOUR_C_THROW("Local ID not found in epetra vector!");

    (*mechanical_stress_state_np_)[doflid_epetra] = (nodal_stresses_source(0))[nodelid];
    (*mechanical_stress_state_np_)[doflid_epetra + 1] = (nodal_stresses_source(1))[nodelid];
    (*mechanical_stress_state_np_)[doflid_epetra + 2] = (nodal_stresses_source(2))[nodelid];
    (*mechanical_stress_state_np_)[doflid_epetra + 3] = (nodal_stresses_source(3))[nodelid];
    (*mechanical_stress_state_np_)[doflid_epetra + 4] = (nodal_stresses_source(4))[nodelid];
    (*mechanical_stress_state_np_)[doflid_epetra + 5] = (nodal_stresses_source(5))[nodelid];
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::LinAlg::Map> Solid::ModelEvaluator::BaseSSI::get_block_dof_row_map_ptr()
    const
{
  check_init_setup();
  return global_state().dof_row_map();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BaseSSI::read_restart(Core::IO::DiscretizationReader& ioreader)
{
  if (mechanical_stress_state_n_ != nullptr)
    ioreader.read_vector(mechanical_stress_state_n_, "stress_n");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BaseSSI::setup()
{
  // check initialization
  check_init();

  if (discret().num_dof_sets() - 1 == 2)
  {
    mechanical_stress_state_n_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret().dof_row_map(2), true);
    mechanical_stress_state_np_ =
        std::make_shared<Core::LinAlg::Vector<double>>(*discret().dof_row_map(2), true);
  }

  // set flag
  issetup_ = true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Solid::ModelEvaluator::BaseSSI::write_restart(
    Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const
{
  if (mechanical_stress_state_n_ != nullptr)
    iowriter.write_vector("stress_n", mechanical_stress_state_n_);
}



FOUR_C_NAMESPACE_CLOSE
