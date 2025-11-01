// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_gauss_point_postprocess.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_post_common.hpp"
#include "4C_post_processor_single_field_writers.hpp"
#include "4C_post_writer_base.hpp"

#include <Teuchos_ParameterList.hpp>

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::post_stress(const std::string groupname, const std::string stresstype)
{
  PostField* field = writer_->get_field();
  PostResult result = PostResult(field);
  result.next_result();

  if (!map_has_map(result.group(), groupname.c_str())) return;

  //--------------------------------------------------------------------
  // calculation and output of nodal stresses in xyz-reference frame
  //--------------------------------------------------------------------

  if (stresstype == "ndxyz")
  {
    write_stress(groupname, result, nodebased);
  }

  //-------------------------------------------------------------------------
  // calculation and output of element center stresses in xyz-reference frame
  //-------------------------------------------------------------------------

  else if (stresstype == "cxyz")
  {
    write_stress(groupname, result, elementbased);
  }

  //-----------------------------------------------------------------------------------
  // calculation and output of nodal and element center stresses in xyz-reference frame
  //-----------------------------------------------------------------------------------

  else if (stresstype == "cxyz_ndxyz")
  {
    write_stress(groupname, result, nodebased);

    // reset result for postprocessing and output of element center stresses
    PostResult resultelestress = PostResult(field);
    resultelestress.next_result();
    write_stress(groupname, resultelestress, elementbased);
  }

  else if (stresstype == "nd123")
  {
    write_eigen_stress(groupname, result, nodebased);
  }

  else if (stresstype == "c123")
  {
    write_eigen_stress(groupname, result, elementbased);
  }

  else if (stresstype == "c123_nd123")
  {
    write_eigen_stress(groupname, result, nodebased);

    // reset result for postprocessing and output of element center stresses
    PostResult resultelestress = PostResult(field);
    resultelestress.next_result();
    write_eigen_stress(groupname, resultelestress, elementbased);
  }

  else
  {
    FOUR_C_THROW("Unknown stress/strain type");
  }

  return;
}



//--------------------------------------------------------------------
// calculate nodal stresses from gauss point stresses
//--------------------------------------------------------------------
struct WriteNodalStressStep : public SpecialFieldInterface
{
  WriteNodalStressStep(StructureFilter& filter) : filter_(filter) {}

  std::vector<int> num_df_map() override { return std::vector<int>(1, 6); }

  void operator()(std::vector<std::shared_ptr<std::ofstream>>& files, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::vector<std::string>& name) override
  {
    using namespace FourC;

    FOUR_C_ASSERT(name.size() == 1, "Unexpected number of names");

    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
        result.read_result_serialdensematrix(groupname);

    const std::shared_ptr<Core::FE::Discretization> dis = result.field()->discretization();
    const Core::LinAlg::Map* noderowmap = dis->node_row_map();

    Teuchos::ParameterList p;
    Core::LinAlg::MultiVector<double> nodal_stress(*noderowmap, 6, true);

    dis->evaluate(
        [&](Core::Elements::Element& ele)
        {
          Core::FE::extrapolate_gauss_point_quantity_to_nodes(
              ele, *data->at(ele.id()), *dis, nodal_stress);
        });

    filter_.get_writer().write_nodal_result_step(*files[0],
        Core::Utils::shared_ptr_from_ref(nodal_stress), resultfilepos, groupname, name[0], 6);
  }

  StructureFilter& filter_;
};



//--------------------------------------------------------------------
// calculate element center stresses from gauss point stresses
//--------------------------------------------------------------------
struct WriteElementCenterStressStep : public SpecialFieldInterface
{
  WriteElementCenterStressStep(StructureFilter& filter) : filter_(filter) {}

  std::vector<int> num_df_map() override { return std::vector<int>(1, 6); }

  void operator()(std::vector<std::shared_ptr<std::ofstream>>& files, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::vector<std::string>& name) override
  {
    using namespace FourC;

    FOUR_C_ASSERT(name.size() == 1, "Unexpected number of names");
    const std::shared_ptr<Core::FE::Discretization> dis = result.field()->discretization();
    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
        result.read_result_serialdensematrix(groupname);

    Core::LinAlg::MultiVector<double> elestress(*(dis->element_row_map()), 6);

    dis->evaluate(
        [&](Core::Elements::Element& ele)
        {
          Core::FE::evaluate_gauss_point_quantity_at_element_center(
              ele, *data->at(ele.id()), elestress);
        });

    filter_.get_writer().write_element_result_step(*files[0],
        Core::Utils::shared_ptr_from_ref(elestress), resultfilepos, groupname, name[0], 6, 0);
  }

  StructureFilter& filter_;
};



//--------------------------------------------------------------------
// Get structural rotation tensor R at element center     pfaller may17
//--------------------------------------------------------------------
struct WriteElementCenterRotation : public SpecialFieldInterface
{
  WriteElementCenterRotation(StructureFilter& filter) : filter_(filter) {}

  std::vector<int> num_df_map() override { return std::vector<int>(1, 9); }

  void operator()(std::vector<std::shared_ptr<std::ofstream>>& files, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::vector<std::string>& name) override
  {
    using namespace FourC;

    FOUR_C_ASSERT(name.size() == 1, "Unexpected number of names");
    const std::shared_ptr<Core::FE::Discretization> dis = result.field()->discretization();
    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
        result.read_result_serialdensematrix(groupname);

    Core::LinAlg::MultiVector<double> elerotation(*(dis->element_row_map()), 9);
    dis->evaluate(
        [&](Core::Elements::Element& ele)
        {
          const Core::LinAlg::SerialDenseMatrix& elecenterrot = *data->at(ele.id());

          const Core::LinAlg::Map& elemap = elerotation.get_map();
          int lid = elemap.lid(ele.id());
          if (lid != -1)
            for (int i = 0; i < elecenterrot.numRows(); ++i)
              for (int j = 0; j < elecenterrot.numCols(); ++j)
                ((elerotation(i * elecenterrot.numRows() + j))).get_values()[lid] =
                    elecenterrot(i, j);
        });

    filter_.get_writer().write_element_result_step(*files[0],
        Core::Utils::shared_ptr_from_ref(elerotation), resultfilepos, groupname, name[0], 9, 0);
  }

  StructureFilter& filter_;
};

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::write_stress(
    const std::string groupname, PostResult& result, const ResultType stresskind)
{
  std::string name;
  std::string out;

  if (groupname == "gauss_2PK_stresses_xyz")
  {
    name = "2PK_stresses_xyz";
    out = "2nd Piola-Kirchhoff stresses";
  }
  else if (groupname == "gauss_cauchy_stresses_xyz")
  {
    name = "cauchy_stresses_xyz";
    out = "Cauchy stresses";
  }
  else if (groupname == "gauss_GL_strains_xyz")
  {
    name = "GL_strains_xyz";
    out = "Green-Lagrange strains";
  }
  else if (groupname == "gauss_EA_strains_xyz")
  {
    name = "EA_strains_xyz";
    out = "Euler-Almansi strains";
  }
  else if (groupname == "gauss_LOG_strains_xyz")
  {
    name = "LOG_strains_xyz";
    out = "Logarithmic strains";
  }
  else if (groupname == "gauss_pl_GL_strains_xyz")
  {
    name = "pl_GL_strains_xyz";
    out = "Plastic Green-Lagrange strains";
  }
  else if (groupname == "gauss_pl_EA_strains_xyz")
  {
    name = "pl_EA_strains_xyz";
    out = "Plastic Euler-Almansi strains";
  }
  else if (groupname == "rotation")
  {
    name = "rotation";
    out = "structural rotation tensor";
  }
  else
  {
    FOUR_C_THROW("trying to write something that is not a stress or a strain");
  }

  if (groupname == "rotation")
  {
    name = "element_" + name;
    WriteElementCenterRotation stresses(*this);
    writer_->write_special_field(
        stresses, result, elementbased, groupname, std::vector<std::string>(1, name), out);
  }

  {
    if (stresskind == nodebased)
    {
      name = "nodal_" + name;
      WriteNodalStressStep stresses(*this);
      writer_->write_special_field(
          stresses, result, nodebased, groupname, std::vector<std::string>(1, name), out);
    }
    else if (stresskind == elementbased)
    {
      name = "element_" + name;
      WriteElementCenterStressStep stresses(*this);
      writer_->write_special_field(
          stresses, result, elementbased, groupname, std::vector<std::string>(1, name), out);
    }
    else
      FOUR_C_THROW("Unknown stress type");
  }
}



//--------------------------------------------------------------------
// calculate nodal eigen stresses from gauss point stresses
//--------------------------------------------------------------------
struct WriteNodalEigenStressStep : public SpecialFieldInterface
{
  WriteNodalEigenStressStep(StructureFilter& filter) : filter_(filter) {}

  std::vector<int> num_df_map() override
  {
    std::vector<int> map(3, 1);
    for (int i = 0; i < 3; ++i) map.push_back(3);
    return map;
  }

  void operator()(std::vector<std::shared_ptr<std::ofstream>>& files, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::vector<std::string>& name) override
  {
    using namespace FourC;

    FOUR_C_ASSERT(name.size() == 6, "Unexpected number of names");

    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
        result.read_result_serialdensematrix(groupname);

    const std::shared_ptr<Core::FE::Discretization> dis = result.field()->discretization();
    const Core::LinAlg::Map* noderowmap = dis->node_row_map();

    Core::LinAlg::MultiVector<double> nodal_stress(*noderowmap, 6, true);

    dis->evaluate(
        [&](Core::Elements::Element& ele)
        {
          Core::FE::extrapolate_gauss_point_quantity_to_nodes(
              ele, *data->at(ele.id()), *dis, nodal_stress);
        });


    // Core::LinAlg::MultiVector<double> with eigenvalues (3) and eigenvectors (9 components) in
    // each row (=node)
    std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>> nodal_eigen_val_vec(6);
    for (int i = 0; i < 3; ++i)
      nodal_eigen_val_vec[i] = std::make_shared<Core::LinAlg::MultiVector<double>>(*noderowmap, 1);
    for (int i = 3; i < 6; ++i)
      nodal_eigen_val_vec[i] = std::make_shared<Core::LinAlg::MultiVector<double>>(*noderowmap, 3);

    const int numnodes = dis->num_my_row_nodes();
    bool threedim = true;
    if (result.field()->problem()->num_dim() == 2) threedim = false;

    // the three-dimensional case
    if (threedim)
    {
      for (int i = 0; i < numnodes; ++i)
      {
        Core::LinAlg::SerialDenseMatrix eigenvec(3, 3);
        Core::LinAlg::SerialDenseVector eigenval(3);

        eigenvec(0, 0) = nodal_stress(0).local_values_as_span()[i];
        eigenvec(0, 1) = nodal_stress(3).local_values_as_span()[i];
        eigenvec(0, 2) = nodal_stress(5).local_values_as_span()[i];
        eigenvec(1, 0) = eigenvec(0, 1);
        eigenvec(1, 1) = nodal_stress(1).local_values_as_span()[i];
        eigenvec(1, 2) = nodal_stress(4).local_values_as_span()[i];
        eigenvec(2, 0) = eigenvec(0, 2);
        eigenvec(2, 1) = eigenvec(1, 2);
        eigenvec(2, 2) = nodal_stress(2).local_values_as_span()[i];

        Core::LinAlg::symmetric_eigen_problem(eigenvec, eigenval, true);

        for (int d = 0; d < 3; ++d)
        {
          (((*nodal_eigen_val_vec[d])(0))).get_values()[i] = eigenval(d);
          for (int e = 0; e < 3; ++e)
            (((*nodal_eigen_val_vec[d + 3])(e))).get_values()[i] = eigenvec(e, d);
        }
      }
    }
    // the two-dimensional case
    else
    {
      for (int i = 0; i < numnodes; ++i)
      {
        Core::LinAlg::SerialDenseMatrix eigenvec(2, 2);
        Core::LinAlg::SerialDenseVector eigenval(2);

        eigenvec(0, 0) = nodal_stress(0).local_values_as_span()[i];
        eigenvec(0, 1) = nodal_stress(3).local_values_as_span()[i];
        eigenvec(1, 0) = eigenvec(0, 1);
        eigenvec(1, 1) = nodal_stress(1).local_values_as_span()[i];

        Core::LinAlg::symmetric_eigen_problem(eigenvec, eigenval, true);

        (((*nodal_eigen_val_vec[0])(0))).get_values()[i] = eigenval(0);
        (((*nodal_eigen_val_vec[1])(0))).get_values()[i] = eigenval(1);
        (((*nodal_eigen_val_vec[2])(0))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[3])(0))).get_values()[i] = eigenvec(0, 0);
        (((*nodal_eigen_val_vec[3])(1))).get_values()[i] = eigenvec(1, 0);
        (((*nodal_eigen_val_vec[3])(2))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[4])(0))).get_values()[i] = eigenvec(0, 1);
        (((*nodal_eigen_val_vec[4])(1))).get_values()[i] = eigenvec(1, 1);
        (((*nodal_eigen_val_vec[4])(2))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(0))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(1))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(2))).get_values()[i] = 0.0;
      }
    }

    for (int i = 0; i < 3; ++i)
      filter_.get_writer().write_nodal_result_step(
          *files[i], nodal_eigen_val_vec[i], resultfilepos, groupname, name[i], 1);
    for (int i = 3; i < 6; ++i)
      filter_.get_writer().write_nodal_result_step(
          *files[i], nodal_eigen_val_vec[i], resultfilepos, groupname, name[i], 3);
  }

  StructureFilter& filter_;
};



//--------------------------------------------------------------------
// calculate element center eigen stresses from gauss point stresses
//--------------------------------------------------------------------
struct WriteElementCenterEigenStressStep : public SpecialFieldInterface
{
  WriteElementCenterEigenStressStep(StructureFilter& filter) : filter_(filter) {}

  std::vector<int> num_df_map() override
  {
    std::vector<int> map(3, 1);
    for (int i = 0; i < 3; ++i) map.push_back(3);
    return map;
  }

  void operator()(std::vector<std::shared_ptr<std::ofstream>>& files, PostResult& result,
      std::map<std::string, std::vector<std::ofstream::pos_type>>& resultfilepos,
      const std::string& groupname, const std::vector<std::string>& name) override
  {
    using namespace FourC;

    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
        result.read_result_serialdensematrix(groupname);

    const std::shared_ptr<Core::FE::Discretization> dis = result.field()->discretization();

    Core::LinAlg::MultiVector<double> element_stress(*dis->element_row_map(), 6, true);

    dis->evaluate(
        [&](Core::Elements::Element& ele)
        {
          Core::FE::evaluate_gauss_point_quantity_at_element_center(
              ele, *data->at(ele.id()), element_stress);
        });


    std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>> nodal_eigen_val_vec(6);
    for (int i = 0; i < 3; ++i)
      nodal_eigen_val_vec[i] =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*(dis->element_row_map()), 1);
    for (int i = 3; i < 6; ++i)
      nodal_eigen_val_vec[i] =
          std::make_shared<Core::LinAlg::MultiVector<double>>(*(dis->element_row_map()), 3);

    const int numnodes = dis->num_my_row_nodes();
    bool threedim = true;
    if (result.field()->problem()->num_dim() == 2) threedim = false;

    // the three-dimensional case
    if (threedim)
    {
      for (int i = 0; i < dis->num_my_row_elements(); ++i)
      {
        Core::LinAlg::SerialDenseMatrix eigenvec(3, 3);
        Core::LinAlg::SerialDenseVector eigenval(3);

        eigenvec(0, 0) = ((element_stress(0))).get_values()[i];
        eigenvec(0, 1) = ((element_stress(3))).get_values()[i];
        eigenvec(0, 2) = ((element_stress(5))).get_values()[i];
        eigenvec(1, 0) = eigenvec(0, 1);
        eigenvec(1, 1) = ((element_stress(1))).get_values()[i];
        eigenvec(1, 2) = ((element_stress(4))).get_values()[i];
        eigenvec(2, 0) = eigenvec(0, 2);
        eigenvec(2, 1) = eigenvec(1, 2);
        eigenvec(2, 2) = ((element_stress(2))).get_values()[i];

        Core::LinAlg::symmetric_eigen_problem(eigenvec, eigenval, true);

        for (int d = 0; d < 3; ++d)
        {
          (((*nodal_eigen_val_vec[d])(0))).get_values()[i] = eigenval(d);
          for (int e = 0; e < 3; ++e)
            (((*nodal_eigen_val_vec[d + 3])(e))).get_values()[i] = eigenvec(e, d);
        }
      }
    }
    // the two-dimensional case
    else
    {
      for (int i = 0; i < numnodes; ++i)
      {
        Core::LinAlg::SerialDenseMatrix eigenvec(2, 2);
        Core::LinAlg::SerialDenseVector eigenval(2);

        eigenvec(0, 0) = element_stress(0).local_values_as_span()[i];
        eigenvec(0, 1) = element_stress(3).local_values_as_span()[i];
        eigenvec(1, 0) = eigenvec(0, 1);
        eigenvec(1, 1) = element_stress(1).local_values_as_span()[i];

        Core::LinAlg::symmetric_eigen_problem(eigenvec, eigenval, true);

        (((*nodal_eigen_val_vec[0])(0))).get_values()[i] = eigenval(0);
        (((*nodal_eigen_val_vec[1])(0))).get_values()[i] = eigenval(1);
        (((*nodal_eigen_val_vec[2])(0))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[3])(0))).get_values()[i] = eigenvec(0, 0);
        (((*nodal_eigen_val_vec[3])(1))).get_values()[i] = eigenvec(1, 0);
        (((*nodal_eigen_val_vec[3])(2))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[4])(0))).get_values()[i] = eigenvec(0, 1);
        (((*nodal_eigen_val_vec[4])(1))).get_values()[i] = eigenvec(1, 1);
        (((*nodal_eigen_val_vec[4])(2))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(0))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(1))).get_values()[i] = 0.0;
        (((*nodal_eigen_val_vec[5])(2))).get_values()[i] = 0.0;
      }
    }

    for (int i = 0; i < 3; ++i)
      filter_.get_writer().write_element_result_step(
          *files[i], nodal_eigen_val_vec[i], resultfilepos, groupname, name[i], 1, 0);
    for (int i = 3; i < 6; ++i)
      filter_.get_writer().write_element_result_step(
          *files[i], nodal_eigen_val_vec[i], resultfilepos, groupname, name[i], 3, 0);
  }

  StructureFilter& filter_;
};



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::write_eigen_stress(
    const std::string groupname, PostResult& result, const ResultType stresskind)
{
  std::vector<std::string> name(6);
  std::string out;

  if (groupname == "gauss_2PK_stresses_xyz")
  {
    name[0] = "2PK_stresses_eigenval1";
    name[1] = "2PK_stresses_eigenval2";
    name[2] = "2PK_stresses_eigenval3";
    name[3] = "2PK_stresses_eigenvec1";
    name[4] = "2PK_stresses_eigenvec2";
    name[5] = "2PK_stresses_eigenvec3";
    out = "principal 2nd Piola-Kirchhoff stresses";
  }
  else if (groupname == "gauss_cauchy_stresses_xyz")
  {
    name[0] = "cauchy_stresses_eigenval1";
    name[1] = "cauchy_stresses_eigenval2";
    name[2] = "cauchy_stresses_eigenval3";
    name[3] = "cauchy_stresses_eigenvec1";
    name[4] = "cauchy_stresses_eigenvec2";
    name[5] = "cauchy_stresses_eigenvec3";
    out = "principal Cauchy stresses";
  }
  else if (groupname == "gauss_GL_strains_xyz")
  {
    name[0] = "GL_strains_eigenval1";
    name[1] = "GL_strains_eigenval2";
    name[2] = "GL_strains_eigenval3";
    name[3] = "GL_strains_eigenvec1";
    name[4] = "GL_strains_eigenvec2";
    name[5] = "GL_strains_eigenvec3";
    out = "principal Green-Lagrange strains";
  }
  else if (groupname == "gauss_EA_strains_xyz")
  {
    name[0] = "EA_strains_eigenval1";
    name[1] = "EA_strains_eigenval2";
    name[2] = "EA_strains_eigenval3";
    name[3] = "EA_strains_eigenvec1";
    name[4] = "EA_strains_eigenvec2";
    name[5] = "EA_strains_eigenvec3";
    out = "principal Euler-Almansi strains";
  }
  else if (groupname == "gauss_LOG_strains_xyz")
  {
    name[0] = "LOG_strains_eigenval1";
    name[1] = "LOG_strains_eigenval2";
    name[2] = "LOG_strains_eigenval3";
    name[3] = "LOG_strains_eigenvec1";
    name[4] = "LOG_strains_eigenvec2";
    name[5] = "LOG_strains_eigenvec3";
    out = "principal Logarithmic strains";
  }
  else if (groupname == "gauss_pl_GL_strains_xyz")
  {
    name[0] = "pl_GL_strains_eigenval1";
    name[1] = "pl_GL_strains_eigenval2";
    name[2] = "pl_GL_strains_eigenval3";
    name[3] = "pl_GL_strains_eigenvec1";
    name[4] = "pl_GL_strains_eigenvec2";
    name[5] = "pl_GL_strains_eigenvec3";
    out = "principal plastic Green-Lagrange strains";
  }
  else if (groupname == "gauss_pl_EA_strains_xyz")
  {
    name[0] = "pl_EA_strains_eigenval1";
    name[1] = "pl_EA_strains_eigenval2";
    name[2] = "pl_EA_strains_eigenval3";
    name[3] = "pl_EA_strains_eigenvec1";
    name[4] = "pl_EA_strains_eigenvec2";
    name[5] = "pl_EA_strains_eigenvec3";
    out = "principal plastic Euler-Almansi strains";
  }
  else
  {
    FOUR_C_THROW("trying to write something that is not a stress or a strain");
  }


  if (stresskind == nodebased)
  {
    for (int i = 0; i < 6; ++i) name[i] = "nodal_" + name[i];
    WriteNodalEigenStressStep stresses(*this);
    writer_->write_special_field(stresses, result, nodebased, groupname, name, out);
  }
  else if (stresskind == elementbased)
  {
    for (int i = 0; i < 6; ++i) name[i] = "element_" + name[i];
    WriteElementCenterEigenStressStep stresses(*this);
    writer_->write_special_field(stresses, result, elementbased, groupname, name, out);
  }
  else
    FOUR_C_THROW("Unknown heatflux type");
}


FOUR_C_NAMESPACE_CLOSE
