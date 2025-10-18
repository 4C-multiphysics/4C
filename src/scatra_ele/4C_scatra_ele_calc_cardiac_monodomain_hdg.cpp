// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_cardiac_monodomain_hdg.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_fiber_node.hpp"
#include "4C_fem_general_fiber_node_holder.hpp"
#include "4C_fem_general_fiber_node_utils.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_myocard.hpp"
#include "4C_scatra_ele_calc_hdg.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype,
    probdim>::ScaTraEleCalcHDGCardiacMonodomain(const int numdofpernode, const int numscal,
    const std::string& disname)
    : Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::ScaTraEleCalcHDG(
          numdofpernode, numscal, disname),
      values_mat_gp_all_(0),
      gp_mat_alpha_(0)
{
}

/*----------------------------------------------------------------------*
 | singleton access method                               hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>*
Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname, bool create)
{
  static std::map<std::string, ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>*> instances;

  if (create)
  {
    if (instances.find(disname) == instances.end())
      instances[disname] =
          new ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>(numdofpernode, numscal, disname);
  }

  else if (instances.find(disname) != instances.end())
  {
    for (typename std::map<std::string,
             ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>*>::iterator i = instances.begin();
        i != instances.end(); ++i)
    {
      delete i->second;
      i->second = nullptr;
    }

    instances.clear();
    return nullptr;
  }

  return instances[disname];
}


/*----------------------------------------------------------------------*
 |  prepare material parameter                           hoermann 11/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::prepare_materials_all(
    Core::Elements::Element* ele,                               //!< the element we are dealing with
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor)
{
  const std::shared_ptr<Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  if (actmat->diffusion_at_ele_center())
  {
    // get diffusivity at ele center
    Core::LinAlg::Matrix<probdim, probdim> diff(Core::LinAlg::Initialization::zero);
    actmat->diffusivity(diff, 0);
    Core::LinAlg::SerialDenseMatrix difftensortmp(this->nsd_, this->nsd_);
    for (unsigned int i = 0; i < this->nsd_; ++i)
      for (unsigned int j = 0; j < this->nsd_; ++j) difftensortmp(i, j) = diff(i, j);
    (*difftensor).push_back(difftensortmp);
    Core::Nodes::FiberNode* fnode = dynamic_cast<Core::Nodes::FiberNode*>(ele->nodes()[0]);
    if (fnode) FOUR_C_THROW("Fiber direction defined twice (nodes and elements)");
  }
  else
  {
    actmat->reset_diffusion_tensor();

    std::shared_ptr<Core::FE::ShapeValues<distype>> shapes =
        std::make_shared<Core::FE::ShapeValues<distype>>(1, false, 2 * hdgele->degree());

    shapes->evaluate(*ele);

    std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>> shapefcns(shapes->nqpoints_);

    for (std::size_t q = 0; q < shapes->nqpoints_; ++q)
    {
      for (std::size_t i = 0; i < shapes->ndofs_; ++i)
      {
        shapefcns[q](i) = shapes->funct(i, q);
      }
    }

    Core::Nodes::NodalFiberHolder gpFiberHolder;
    Core::Nodes::project_fibers_to_gauss_points<distype>(ele->nodes(), shapefcns, gpFiberHolder);

    std::vector<Core::LinAlg::Tensor<double, probdim>> fibergp(shapes->nqpoints_);
    setup_cardiac_fibers<probdim>(gpFiberHolder, fibergp);

    for (unsigned int q = 0; q < shapes->nqpoints_; ++q) actmat->setup_diffusion_tensor(fibergp[q]);

    for (unsigned int q = 0; q < shapes->nqpoints_; ++q)
    {
      Core::LinAlg::SerialDenseMatrix difftensortmp(this->nsd_, this->nsd_);
      Core::LinAlg::Matrix<probdim, probdim> diff(Core::LinAlg::Initialization::zero);
      actmat->diffusivity(diff, q);
      for (unsigned int i = 0; i < this->nsd_; ++i)
        for (unsigned int j = 0; j < this->nsd_; ++j) difftensortmp(i, j) = diff(i, j);
      (*difftensor).push_back(difftensortmp);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 |  prepare material parameter                           hoermann 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::prepare_materials(
    Core::Elements::Element* ele,                               //!< the element we are dealing with
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor)
{
  if (distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
    prepare_materials_tet(ele, material, k, difftensor);
  else
    prepare_materials_all(ele, material, k, difftensor);

  return;
}

/*----------------------------------------------------------------------*
 |  prepare material parameter                           hoermann 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::prepare_materials_tet(
    Core::Elements::Element* ele,                               //!< the element we are dealing with
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor)
{
  const std::shared_ptr<Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  if (actmat->diffusion_at_ele_center())
  {
    // get diffusivity at ele center
    Core::LinAlg::Matrix<probdim, probdim> diff(Core::LinAlg::Initialization::zero);
    actmat->diffusivity(diff, 0);
    Core::LinAlg::SerialDenseMatrix difftensortmp(this->nsd_, this->nsd_);
    for (unsigned int i = 0; i < this->nsd_; ++i)
      for (unsigned int j = 0; j < this->nsd_; ++j) difftensortmp(i, j) = diff(i, j);
    (*difftensor).push_back(difftensortmp);
    Core::Nodes::FiberNode* fnode = dynamic_cast<Core::Nodes::FiberNode*>(ele->nodes()[0]);
    if (fnode) FOUR_C_THROW("Fiber direction defined twice (nodes and elements)");
  }
  else
  {
    actmat->reset_diffusion_tensor();

    const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints(
        ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(2 * hdgele->degree()));
    const std::size_t numgp = intpoints.ip().nquad;

    std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes(distype), 1>> shapefcns(numgp);

    Core::LinAlg::Matrix<probdim, 1> gp_coord(Core::LinAlg::Initialization::zero);
    for (std::size_t q = 0; q < numgp; ++q)
    {
      // gaussian points coordinates
      for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
        gp_coord(idim) = intpoints.ip().qxg[q][idim];

      Core::FE::shape_function<distype>(gp_coord, shapefcns[q]);
    }

    Core::Nodes::NodalFiberHolder gpFiberHolder;
    Core::Nodes::project_fibers_to_gauss_points<distype>(ele->nodes(), shapefcns, gpFiberHolder);

    std::vector<Core::LinAlg::Tensor<double, probdim>> fibergp(numgp);
    setup_cardiac_fibers<probdim>(gpFiberHolder, fibergp);

    for (unsigned int q = 0; q < numgp; ++q) actmat->setup_diffusion_tensor(fibergp[q]);

    for (unsigned int q = 0; q < numgp; ++q)
    {
      Core::LinAlg::SerialDenseMatrix difftensortmp(this->nsd_, this->nsd_);
      Core::LinAlg::Matrix<probdim, probdim> diff(Core::LinAlg::Initialization::zero);
      actmat->diffusivity(diff, q);
      for (unsigned int i = 0; i < this->nsd_; ++i)
        for (unsigned int j = 0; j < this->nsd_; ++j) difftensortmp(i, j) = diff(i, j);
      (*difftensor).push_back(difftensortmp);
    }
  }

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::materials(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    Core::LinAlg::SerialDenseMatrix& difftensor, Core::LinAlg::SerialDenseVector& ivecn,
    Core::LinAlg::SerialDenseVector& ivecnp, Core::LinAlg::SerialDenseMatrix& ivecnpderiv)
{
  if (material->material_type() == Core::Materials::m_myocard)
    mat_myocard(material, k, difftensor, ivecn, ivecnp, ivecnpderiv);
  else
    FOUR_C_THROW("Material type is not supported");

  return;
}


/*----------------------------------------------------------------------*
 |  Material ScaTra                                      hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::mat_myocard(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    Core::LinAlg::SerialDenseMatrix& difftensor, Core::LinAlg::SerialDenseVector& ivecn,
    Core::LinAlg::SerialDenseVector& ivecnp, Core::LinAlg::SerialDenseMatrix& ivecnpderiv)
{
  const std::shared_ptr<const Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<const Mat::Myocard>(material);

  // coordinate of material gauss points
  Core::LinAlg::Matrix<probdim, 1> mat_gp_coord(Core::LinAlg::Initialization::zero);
  // values of shape function at material gauss points
  Core::LinAlg::SerialDenseVector values_mat_gp(this->shapes_->ndofs_);

  double imatgpnpderiv(0.);
  double imatgpnp(0.);
  double imatgpn(0.);

  ivecn.putScalar(0.0);
  ivecnp.putScalar(0.0);
  ivecnpderiv.putScalar(0.0);

  // polynomial space to get the value of the shape function at the material gauss points
  Core::FE::PolynomialSpaceParams params(distype, this->shapes_->degree_, this->usescompletepoly_);
  polySpace_ = Core::FE::PolynomialSpaceCache<probdim>::instance().create(params);

  int nqpoints;

  if (distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
  {
    int deg = 0;
    if (this->shapes_->degree_ == 1)
      deg = 4 * this->shapes_->degree_;
    else
      deg = 3 * this->shapes_->degree_;
    const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints(
        ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(deg));
    nqpoints = intpoints.ip().nquad;

    if (nqpoints != actmat->get_number_of_gp())
      FOUR_C_THROW(
          "Number of quadrature points ({}) does not match number of points in material ({})!",
          nqpoints, actmat->get_number_of_gp());

    if (values_mat_gp_all_.empty() or
        values_mat_gp_all_.size() != (unsigned)actmat->get_number_of_gp())
    {
      values_mat_gp_all_.resize(actmat->get_number_of_gp());
      gp_mat_alpha_.resize(actmat->get_number_of_gp());
    }

    if (unsigned(values_mat_gp_all_[0].numRows()) != this->shapes_->ndofs_)
    {
      for (int q = 0; q < nqpoints; ++q)
      {
        values_mat_gp_all_[q].size(this->shapes_->ndofs_);

        gp_mat_alpha_[q] = intpoints.ip().qwgt[q];
        // gaussian points coordinates
        for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
          mat_gp_coord(idim) = intpoints.ip().qxg[q][idim];

        polySpace_->evaluate(mat_gp_coord, values_mat_gp_all_[q]);
      }
    }
  }
  else
  {
    int deg = 0;
    if (this->shapes_->degree_ == 1)
      deg = 4 * this->shapes_->degree_;
    else
      deg = 3 * this->shapes_->degree_;

    std::shared_ptr<Core::FE::GaussPoints> quadrature_(Core::FE::create_gauss_points(distype, deg));
    nqpoints = quadrature_->num_points();

    if (nqpoints != actmat->get_number_of_gp())
      FOUR_C_THROW(
          "Number of quadrature points ({}) does not match number of points in material ({})!",
          nqpoints, actmat->get_number_of_gp());

    if (values_mat_gp_all_.empty() or
        values_mat_gp_all_.size() != (unsigned)actmat->get_number_of_gp())
    {
      values_mat_gp_all_.resize(actmat->get_number_of_gp());
      gp_mat_alpha_.resize(actmat->get_number_of_gp());
    }

    if (unsigned(values_mat_gp_all_[0].numRows()) != this->shapes_->ndofs_)
    {
      for (int q = 0; q < nqpoints; ++q)
      {
        values_mat_gp_all_[q].size(this->shapes_->ndofs_);

        gp_mat_alpha_[q] = quadrature_->weight(q);
        // gaussian points coordinates
        for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
          mat_gp_coord(idim) = quadrature_->point(q)[idim];

        polySpace_->evaluate(mat_gp_coord, values_mat_gp_all_[q]);
      }
    }
  }

  // Jacobian determinant
  double jacdet = this->shapes_->xjm.determinant();

  for (int q = 0; q < nqpoints; ++q)
  {
    double phinpgp = 0.0;
    double phingp = 0.0;

    // loop over shape functions
    for (unsigned int i = 0; i < this->shapes_->ndofs_; ++i)
    {
      phingp += values_mat_gp_all_[q](i) * this->interiorPhin_(i);
      phinpgp += values_mat_gp_all_[q](i) * this->interiorPhinp_(i);
    }

    // Reaction term at material gauss points

    if (!this->scatrapara_->semi_implicit())
    {
      imatgpnpderiv = actmat->rea_coeff_deriv(phinpgp, this->dt(), q);
    }

    imatgpn = actmat->rea_coeff_n(phingp, this->dt(), q);
    imatgpnp = actmat->rea_coeff(phinpgp, this->dt(), q);

    // loop over shape functions
    for (unsigned int i = 0; i < this->shapes_->ndofs_; ++i)
    {
      ivecn(i) += imatgpn * values_mat_gp_all_[q](i) * jacdet * gp_mat_alpha_[q];
    }

    if (!this->scatrapara_->semi_implicit())
      for (unsigned int i = 0; i < this->shapes_->ndofs_; ++i)
      {
        for (unsigned int j = 0; j < this->shapes_->ndofs_; ++j)
          ivecnpderiv(i, j) += imatgpnpderiv * values_mat_gp_all_[q](i) * values_mat_gp_all_[q](j) *
                               jacdet * gp_mat_alpha_[q];
        ivecnp(i) += imatgpnp * values_mat_gp_all_[q](i) * jacdet * gp_mat_alpha_[q];
      }
  }

  return;
}  // ScaTraEleCalcHDGCardiacMonodomain<distype>::MatMyocard


/*----------------------------------------------------------------------*
 |  Material Time Update                                 hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::time_update_material(
    const Core::Elements::Element* ele  //!< the element we are dealing with
)
{
  std::vector<std::shared_ptr<Mat::Myocard>> updatemat;

  // access the general material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  // first, determine the materials which need a time update, i.e. myocard materials
  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<Mat::MatList> actmat = std::dynamic_pointer_cast<Mat::MatList>(material);
    if (actmat->num_mat() < this->numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < this->numscal_; ++k)
    {
      const int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      if (singlemat->material_type() == Core::Materials::m_myocard)
      {
        // reference to Teuchos::rcp not possible here, since the material
        // is required to be not const for this application
        updatemat.push_back(std::dynamic_pointer_cast<Mat::Myocard>(singlemat));
      }
    }
  }

  if (material->material_type() == Core::Materials::m_myocard)
  {
    // reference to Teuchos::rcp not possible here, since the material is required to be
    // not const for this application
    updatemat.push_back(std::dynamic_pointer_cast<Mat::Myocard>(material));
  }

  if (updatemat.size() > 0)  // found at least one material to be updated
  {
    for (unsigned i = 0; i < updatemat.size(); i++) updatemat[i]->update(0.0, 0.0);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Get Material Internal State for Restart              hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype,
    probdim>::get_material_internal_state(const Core::Elements::Element*
                                              ele,  //!< the element we are dealing with
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization)
{
  // NOTE: add integral values only for elements which are NOT ghosted!
  if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
  {
    // access the general material
    std::shared_ptr<Core::Mat::Material> material = ele->material();
    std::shared_ptr<Core::LinAlg::MultiVector<double>> material_internal_state =
        params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("material_internal_state");

    if (material->material_type() == Core::Materials::m_myocard)
    {
      std::shared_ptr<Mat::Myocard> material =
          std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
      for (int k = 0; k < material->get_number_of_internal_state_variables(); ++k)
      {
        double material_state = 0;
        unsigned int nqpoints = material->get_number_of_gp();
        for (unsigned int q = 0; q < nqpoints; ++q)
        {
          material_state += material->get_internal_state(k, q);
        }
        material_internal_state->replace_global_value(ele->id(), k, material_state / nqpoints);
      }
    }

    params.set<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(
        "material_internal_state", material_internal_state);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  Set Material Internal State after Restart            hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype,
    probdim>::set_material_internal_state(const Core::Elements::Element*
                                              ele,  //!< the element we are dealing with
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization)
{
  // NOTE: add integral values only for elements which are NOT ghosted!
  if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
  {
    // access the general material
    std::shared_ptr<Core::Mat::Material> material = ele->material();
    std::shared_ptr<Core::LinAlg::MultiVector<double>> material_internal_state =
        params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("material_internal_state");

    if (material->material_type() == Core::Materials::m_myocard)
    {
      std::shared_ptr<Mat::Myocard> material =
          std::dynamic_pointer_cast<Mat::Myocard>(ele->material());
      for (int k = 0; k < material->get_number_of_internal_state_variables(); ++k)
      {
        int nqpoints = material->get_number_of_gp();
        for (int q = 0; q < nqpoints; ++q)
        {
          auto material_internal_state_component =
              Core::Utils::shared_ptr_from_ref((*material_internal_state)(k * nqpoints + q));
          material->set_internal_state(k, (*material_internal_state_component)[ele->id()], q);
        }
      }
    }
  }


  return;
}

/*----------------------------------------------------------------------*
 |  Project Material Field                               hoermann 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::project_material_field(
    const Core::Elements::Element* ele  //!< the element we are dealing with
)
{
  if (distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
    return project_material_field_tet(ele);
  else
    return project_material_field_all(ele);
}


/*----------------------------------------------------------------------*
 |  Project Material Field                               hoermann 12/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype,
    probdim>::project_material_field_all(const Core::Elements::Element*
        ele  //!< the element we are dealing with
)
{
  const std::shared_ptr<Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<Mat::Myocard>(ele->material());

  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  int deg = 0;
  if (hdgele->degree() == 1)
    deg = 4 * hdgele->degree();
  else
    deg = 3 * hdgele->degree();
  int degold = 0;
  if (hdgele->degree_old() == 1)
    degold = 4 * hdgele->degree_old();
  else
    degold = 3 * hdgele->degree_old();

  std::shared_ptr<Core::FE::ShapeValues<distype>> shapes =
      std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgele->degree_old(), this->usescompletepoly_, deg);

  std::shared_ptr<Core::FE::ShapeValues<distype>> shapes_old =
      std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgele->degree_old(), this->usescompletepoly_, degold);

  shapes->evaluate(*ele);
  shapes_old->evaluate(*ele);

  Core::LinAlg::SerialDenseMatrix massPartOld(shapes->ndofs_, shapes_old->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPartOldW(shapes->ndofs_, shapes_old->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPart(shapes->ndofs_, shapes->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPartW(shapes->ndofs_, shapes->nqpoints_);
  Core::LinAlg::SerialDenseMatrix Mmat(shapes->ndofs_, shapes->ndofs_);

  Core::LinAlg::SerialDenseMatrix state_variables(
      shapes_old->nqpoints_, actmat->get_number_of_internal_state_variables());

  if (shapes->ndofs_ != shapes_old->ndofs_)
    FOUR_C_THROW("Number of shape functions not identical!");

  for (unsigned int i = 0; i < shapes->ndofs_; ++i)
  {
    for (unsigned int q = 0; q < shapes->nqpoints_; ++q)
    {
      massPart(i, q) = shapes->shfunct(i, q);
      massPartW(i, q) = shapes->shfunct(i, q) * shapes->jfac(q);
    }
    for (unsigned int q = 0; q < shapes_old->nqpoints_; ++q)
    {
      massPartOld(i, q) = shapes_old->shfunct(i, q);
      massPartOldW(i, q) = shapes_old->shfunct(i, q) * shapes_old->jfac(q);
    }
  }

  Core::LinAlg::multiply_nt(Mmat, massPartOld, massPartOldW);

  for (unsigned int q = 0; q < shapes_old->nqpoints_; ++q)
    for (int k = 0; k < actmat->get_number_of_internal_state_variables(); ++k)
      state_variables(q, k) = actmat->get_internal_state(k, q);

  Core::LinAlg::SerialDenseMatrix tempMat1(
      shapes->ndofs_, actmat->get_number_of_internal_state_variables());
  Core::LinAlg::multiply(tempMat1, massPartOldW, state_variables);

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMat;
  inverseMat.setMatrix(Teuchos::rcpFromRef(Mmat.base()));
  inverseMat.setVectors(Teuchos::rcpFromRef(tempMat1.base()), Teuchos::rcpFromRef(tempMat1.base()));
  inverseMat.factorWithEquilibration(true);
  int err2 = inverseMat.factor();
  int err = inverseMat.solve();
  if (err != 0 || err2 != 0) FOUR_C_THROW("Inversion of matrix failed with errorcode {}", err);

  Core::LinAlg::SerialDenseMatrix tempMat2(
      shapes->nqpoints_, actmat->get_number_of_internal_state_variables());
  Core::LinAlg::multiply_tn(tempMat2, massPart, tempMat1);

  actmat->set_gp(shapes->nqpoints_);
  actmat->resize_internal_state_variables();


  for (unsigned int q = 0; q < shapes->nqpoints_; ++q)
    for (int k = 0; k < actmat->get_number_of_internal_state_variables(); ++k)
      actmat->set_internal_state(k, tempMat2(q, k), q);

  return 0;
}


/*----------------------------------------------------------------------*
 |  Project Material Field for Tet                       hoermann 01/17 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype,
    probdim>::project_material_field_tet(const Core::Elements::Element*
        ele  //!< the element we are dealing with
)
{
  const std::shared_ptr<Mat::Myocard>& actmat =
      std::dynamic_pointer_cast<Mat::Myocard>(ele->material());

  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  // polynomial space to get the value of the shape function at the material gauss points
  Core::FE::PolynomialSpaceParams params(distype, hdgele->degree_old(), this->usescompletepoly_);
  std::shared_ptr<Core::FE::PolynomialSpace<probdim>> polySpace =
      Core::FE::PolynomialSpaceCache<probdim>::instance().create(params);

  int deg = 0;
  int degold = 0;

  if (hdgele->degree() == 1)
    deg = 4 * hdgele->degree();
  else
    deg = 3 * hdgele->degree();

  if (hdgele->degree_old() == 1)
    degold = 4 * hdgele->degree_old();
  else
    degold = 3 * hdgele->degree_old();

  const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints_old(
      ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(degold));
  const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints(
      ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(deg));


  std::vector<Core::LinAlg::SerialDenseVector> shape_gp_old(intpoints_old.ip().nquad);
  std::vector<Core::LinAlg::SerialDenseVector> shape_gp(intpoints.ip().nquad);

  // coordinate of material gauss points
  Core::LinAlg::Matrix<probdim, 1> mat_gp_coord(Core::LinAlg::Initialization::zero);

  for (int q = 0; q < intpoints_old.ip().nquad; ++q)
  {
    shape_gp_old[q].size(polySpace->size());

    // gaussian points coordinates
    for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
      mat_gp_coord(idim) = intpoints_old.ip().qxg[q][idim];
    polySpace->evaluate(mat_gp_coord, shape_gp_old[q]);
  }

  for (int q = 0; q < intpoints.ip().nquad; ++q)
  {
    shape_gp[q].size(polySpace->size());

    // gaussian points coordinates
    for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
      mat_gp_coord(idim) = intpoints.ip().qxg[q][idim];
    polySpace->evaluate(mat_gp_coord, shape_gp[q]);
  }

  this->shapes_->evaluate(*ele);
  // Jacobian determinant
  double jacdet = this->shapes_->xjm.determinant();



  Core::LinAlg::SerialDenseMatrix massPartOld(polySpace->size(), shape_gp_old.size());
  Core::LinAlg::SerialDenseMatrix massPartOldW(polySpace->size(), shape_gp_old.size());
  Core::LinAlg::SerialDenseMatrix massPart(polySpace->size(), shape_gp.size());
  Core::LinAlg::SerialDenseMatrix massPartW(polySpace->size(), shape_gp.size());
  Core::LinAlg::SerialDenseMatrix Mmat(polySpace->size(), polySpace->size());

  Core::LinAlg::SerialDenseMatrix state_variables(
      shape_gp_old.size(), actmat->get_number_of_internal_state_variables());

  for (unsigned int i = 0; i < polySpace->size(); ++i)
  {
    for (unsigned int q = 0; q < shape_gp.size(); ++q)
    {
      massPart(i, q) = shape_gp[q](i);
      massPartW(i, q) = shape_gp[q](i) * jacdet * intpoints.ip().qwgt[q];
    }
    for (unsigned int q = 0; q < shape_gp_old.size(); ++q)
    {
      massPartOld(i, q) = shape_gp_old[q](i);
      massPartOldW(i, q) = shape_gp_old[q](i) * jacdet * intpoints_old.ip().qwgt[q];
    }
  }

  Core::LinAlg::multiply_nt(Mmat, massPartOld, massPartOldW);

  for (unsigned int q = 0; q < shape_gp_old.size(); ++q)
    for (int k = 0; k < actmat->get_number_of_internal_state_variables(); ++k)
      state_variables(q, k) = actmat->get_internal_state(k, q);

  Core::LinAlg::SerialDenseMatrix tempMat1(
      polySpace->size(), actmat->get_number_of_internal_state_variables());
  Core::LinAlg::multiply(tempMat1, massPartOldW, state_variables);

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMat;
  inverseMat.setMatrix(Teuchos::rcpFromRef(Mmat.base()));
  inverseMat.setVectors(Teuchos::rcpFromRef(tempMat1.base()), Teuchos::rcpFromRef(tempMat1.base()));
  inverseMat.factorWithEquilibration(true);
  int err2 = inverseMat.factor();
  int err = inverseMat.solve();
  if (err != 0 || err2 != 0) FOUR_C_THROW("Inversion of matrix failed with errorcode {}", err);

  Core::LinAlg::SerialDenseMatrix tempMat2(
      shape_gp.size(), actmat->get_number_of_internal_state_variables());
  Core::LinAlg::multiply_tn(tempMat2, massPart, tempMat1);

  actmat->set_gp(shape_gp.size());
  actmat->resize_internal_state_variables();


  for (unsigned int q = 0; q < shape_gp.size(); ++q)
    for (int k = 0; k < actmat->get_number_of_internal_state_variables(); ++k)
      actmat->set_internal_state(k, tempMat2(q, k), q);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/

template <Core::FE::CellType distype, int probdim>
template <std::size_t dim>
void Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<distype, probdim>::setup_cardiac_fibers(
    const Core::Nodes::NodalFiberHolder& fibers, std::vector<Core::LinAlg::Tensor<double, dim>>& f)
{
  if (fibers.fibers_size() > 0)
  {
    const std::vector<Core::LinAlg::Tensor<double, 3>>& fib = fibers.get_fiber(0);
    f.resize(fib.size());
    for (std::size_t gp = 0; gp < fib.size(); ++gp)
    {
      for (std::size_t i = 0; i < dim; ++i)
      {
        f[gp](i) = fib[gp](i);
      }
    }
  }
  else if (fibers.contains_coordinate_system_direction(
               Core::Nodes::CoordinateSystemDirection::Circular) &&
           fibers.contains_coordinate_system_direction(
               Core::Nodes::CoordinateSystemDirection::Tangential))
  {
    const std::vector<Core::LinAlg::Tensor<double, 3>>& cir =
        fibers.get_coordinate_system_direction(Core::Nodes::CoordinateSystemDirection::Circular);
    const std::vector<Core::LinAlg::Tensor<double, 3>>& tan =
        fibers.get_coordinate_system_direction(Core::Nodes::CoordinateSystemDirection::Tangential);
    const std::vector<double>& helix = fibers.get_angle(Core::Nodes::AngleType::Helix);
    const std::vector<double>& transverse = fibers.get_angle(Core::Nodes::AngleType::Transverse);
    f.resize(cir.size());

    double deg2rad = std::numbers::pi / 180.;
    for (unsigned int gp = 0; gp < cir.size(); ++gp)
    {
      Core::LinAlg::Matrix<3, 1> rad(Core::LinAlg::Initialization::uninitialized);
      rad.cross_product(Core::LinAlg::make_matrix_view<3, 1>(cir[gp]),
          Core::LinAlg::make_matrix_view<3, 1>(tan[gp]));

      double tmp1 = cos(helix[gp] * deg2rad) * cos(transverse[gp] * deg2rad);
      double tmp2 = sin(helix[gp] * deg2rad) * cos(transverse[gp] * deg2rad);
      double tmp3 = sin(transverse[gp] * deg2rad);

      for (unsigned int i = 0; i < dim; ++i)
      {
        f[gp](i) = tmp1 * cir[gp](i) + tmp2 * tan[gp](i) + tmp3 * rad(i);
      }
      f[gp] *= 1.0 / Core::LinAlg::norm2(f[gp]);
    }
  }
  else
  {
    FOUR_C_THROW("You have to specify either FIBER1 or CIR, TAN, HELIX and TRANS");
  }
}



// template classes
// 1D elements
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::line2,1>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::line2,2>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::line2,3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::line3,1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::tri3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::quad4, 3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::hex8, 3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::tet10, 3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::pyramid5,
    3>;
// template class
// Discret::Elements::ScaTraEleCalcHDGCardiacMonodomain<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
