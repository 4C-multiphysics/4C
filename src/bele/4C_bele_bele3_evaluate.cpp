// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_bele_bele3.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            gammi 04/07|
 *----------------------------------------------------------------------*/
int Discret::Elements::Bele3::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // start with "none"
  Discret::Elements::Bele3::ActionType act = Bele3::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "calc_struct_constrvol")
    act = Bele3::calc_struct_constrvol;
  else if (action == "calc_struct_volconstrstiff")
    act = Bele3::calc_struct_volconstrstiff;
  else if (action == "calc_struct_stress")
    act = Bele3::calc_struct_stress;

  // what the element has to do
  switch (act)
  {
    // BELE speciality: element action not implemented -> do nothing
    case none:
      break;
    case calc_struct_stress:
    {
      std::shared_ptr<std::vector<char>> stressdata =
          params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
      std::shared_ptr<std::vector<char>> straindata =
          params.get<std::shared_ptr<std::vector<char>>>("strain", nullptr);

      // dummy size for stress/strain. size does not matter. just write something that can be
      // extracted later
      Core::LinAlg::Matrix<1, 1> dummy(Core::LinAlg::Initialization::zero);

      // write dummy stress
      {
        Core::Communication::PackBuffer data;
        add_to_pack(data, dummy);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }

      // write dummy strain
      {
        Core::Communication::PackBuffer data;
        add_to_pack(data, dummy);
        std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
      }
    }
    break;
    case calc_struct_constrvol:
    {
      // create communicator
      MPI_Comm Comm = discretization.get_comm();

      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(Comm) == owner())
      {
        // element geometry update
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement");
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
        std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
        const int numdim = 3;
        Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
        spatial_configuration(xscurr, mydisp);
        // call submethod for volume evaluation and store rseult in third systemvector
        double volumeele = compute_constr_vols(xscurr, num_node());
        elevec3[0] = volumeele;
      }
    }
    break;
    case calc_struct_volconstrstiff:
    {
      // element geometry update
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
      const int numdim = 3;
      Core::LinAlg::SerialDenseMatrix xscurr(num_node(), numdim);  // material coord. of element
      spatial_configuration(xscurr, mydisp);
      double volumeele;
      // first partial derivatives
      Core::LinAlg::SerialDenseVector Vdiff1;
      // second partial derivatives
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Vdiff2 =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>();

      // get projection method
      std::shared_ptr<Core::Conditions::Condition> condition =
          params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
      const std::string* projtype = condition->parameters().get_if<std::string>("projection");

      if (projtype != nullptr)
      {
        // call submethod to compute volume and its derivatives w.r.t. to current displ.
        if (*projtype == "yz")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 0, 0);
        }
        else if (*projtype == "xz")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 1, 1);
        }
        else if (*projtype == "xy")
        {
          compute_vol_deriv(
              xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2, 2, 2);
        }
        else
        {
          compute_vol_deriv(xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2);
        }
      }
      else
        compute_vol_deriv(xscurr, num_node(), numdim * num_node(), volumeele, Vdiff1, Vdiff2);

      // update rhs vector and corresponding column in "constraint" matrix
      elevec1 = Vdiff1;
      elevec2 = Vdiff1;
      elemat1 = *Vdiff2;
      // call submethod for volume evaluation and store result in third systemvector
      elevec3[0] = volumeele;
    }
    break;
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  do nothing (public)                                      a.ger 07/07|
 |                                                                      |
 |  The function is just a dummy.                                       |
 *----------------------------------------------------------------------*/
int Discret::Elements::Bele3::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}


/*----------------------------------------------------------------------*
 * Compute Volume enclosed by surface.                          tk 10/07*
 * ---------------------------------------------------------------------*/
double Discret::Elements::Bele3::compute_constr_vols(
    const Core::LinAlg::SerialDenseMatrix& xc, const int numnode)
{
  double V = 0.0;

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = 0; indc < 3; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    Core::LinAlg::SerialDenseMatrix ab = xc;
    Core::LinAlg::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const Core::FE::IntegrationPoints2D intpoints(get_optimal_gaussrule());
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::SerialDenseVector funct(numnode);
    Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      Core::FE::shape_function_2d(funct, e0, e1, shape());
      Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

      double detA;
      // compute "metric tensor" deriv*ab, which is a 2x3 matrix with zero indc'th column
      Core::LinAlg::SerialDenseMatrix metrictensor(2, 3);
      Core::LinAlg::multiply(metrictensor, deriv, ab);
      // Core::LinAlg::SerialDenseMatrix metrictensor(2,2);
      // metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA = metrictensor(0, inda) * metrictensor(1, indb) -
             metrictensor(0, indb) * metrictensor(1, inda);
      const double dotprodc = funct.dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc * detA * intpoints.qwgt[gpid];
    }
  }
  return V / 3.0;
}

/*----------------------------------------------------------------------*
 * Compute volume and its first and second derivatives          tk 02/09*
 * with respect to the displacements                                    *
 * ---------------------------------------------------------------------*/
void Discret::Elements::Bele3::compute_vol_deriv(const Core::LinAlg::SerialDenseMatrix& xc,
    const int numnode, const int ndof, double& V, Core::LinAlg::SerialDenseVector& Vdiff1,
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> Vdiff2, const int minindex, const int maxindex)
{
  // necessary constants
  const int numdim = 3;
  const double invnumind = 1.0 / (maxindex - minindex + 1.0);

  // initialize
  V = 0.0;
  Vdiff1.size(ndof);
  if (Vdiff2 != nullptr) Vdiff2->shape(ndof, ndof);

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = minindex; indc < maxindex + 1; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    Core::LinAlg::SerialDenseMatrix ab = xc;
    Core::LinAlg::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const Core::FE::IntegrationPoints2D intpoints(get_optimal_gaussrule());
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::SerialDenseVector funct(numnode);
    Core::LinAlg::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      Core::FE::shape_function_2d(funct, e0, e1, shape());
      Core::FE::shape_function_2d_deriv1(deriv, e0, e1, shape());

      // evaluate Jacobi determinant, for projected dA*
      std::vector<double> normal(numdim);
      double detA;
      // compute "metric tensor" deriv*xy, which is a 2x3 matrix with zero 3rd column
      Core::LinAlg::SerialDenseMatrix metrictensor(2, numdim);
      Core::LinAlg::multiply(metrictensor, deriv, ab);
      // metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA = metrictensor(0, inda) * metrictensor(1, indb) -
             metrictensor(0, indb) * metrictensor(1, inda);
      const double dotprodc = funct.dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc * detA * intpoints.qwgt[gpid];

      //-------- compute first derivative
      for (int i = 0; i < numnode; i++)
      {
        (Vdiff1)[3 * i + inda] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(0, i) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, i));
        (Vdiff1)[3 * i + indb] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(1, i) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, i));
        (Vdiff1)[3 * i + indc] += invnumind * intpoints.qwgt[gpid] * funct[i] * detA;
      }

      //-------- compute second derivative
      if (Vdiff2 != nullptr)
      {
        for (int i = 0; i < numnode; i++)
        {
          for (int j = 0; j < numnode; j++)
          {
            //"diagonal" (dV)^2/(dx_i dx_j) = 0, therefore only six entries have to be specified
            (*Vdiff2)(3 * i + inda, 3 * j + indb) +=
                invnumind * intpoints.qwgt[gpid] * dotprodc *
                (deriv(0, i) * deriv(1, j) - deriv(1, i) * deriv(0, j));
            (*Vdiff2)(3 * i + indb, 3 * j + inda) +=
                invnumind * intpoints.qwgt[gpid] * dotprodc *
                (deriv(0, j) * deriv(1, i) - deriv(1, j) * deriv(0, i));
            (*Vdiff2)(3 * i + inda, 3 * j + indc) +=
                invnumind * intpoints.qwgt[gpid] * funct[j] *
                (deriv(0, i) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, i));
            (*Vdiff2)(3 * i + indc, 3 * j + inda) +=
                invnumind * intpoints.qwgt[gpid] * funct[i] *
                (deriv(0, j) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, j));
            (*Vdiff2)(3 * i + indb, 3 * j + indc) +=
                invnumind * intpoints.qwgt[gpid] * funct[j] *
                (deriv(1, i) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, i));
            (*Vdiff2)(3 * i + indc, 3 * j + indb) +=
                invnumind * intpoints.qwgt[gpid] * funct[i] *
                (deriv(1, j) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, j));
          }
        }
      }
    }
  }
  V *= invnumind;
  return;
}

FOUR_C_NAMESPACE_CLOSE
