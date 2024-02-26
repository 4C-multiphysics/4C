/*----------------------------------------------------------------------*/
/*! \file

\brief dummy 3D boundary element without any physics


\level 2
*/
/*----------------------------------------------------------------------*/

#include "baci_bele_bele3.hpp"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.hpp"
#include "baci_lib_discret.hpp"
#include "baci_lib_utils.hpp"
#include "baci_linalg_serialdensematrix.hpp"
#include "baci_linalg_serialdensevector.hpp"
#include "baci_linalg_utils_densematrix_multiply.hpp"
#include "baci_linalg_utils_sparse_algebra_math.hpp"
#include "baci_mat_newtonianfluid.hpp"
#include "baci_utils_exceptions.hpp"

BACI_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 |  evaluate the element (public)                            gammi 04/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Bele3::Evaluate(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, std::vector<int>& lm,
    CORE::LINALG::SerialDenseMatrix& elemat1, CORE::LINALG::SerialDenseMatrix& elemat2,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseVector& elevec2,
    CORE::LINALG::SerialDenseVector& elevec3)
{
  // start with "none"
  DRT::ELEMENTS::Bele3::ActionType act = Bele3::none;

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
      Teuchos::RCP<std::vector<char>> stressdata =
          params.get<Teuchos::RCP<std::vector<char>>>("stress", Teuchos::null);
      Teuchos::RCP<std::vector<char>> straindata =
          params.get<Teuchos::RCP<std::vector<char>>>("strain", Teuchos::null);

      // dummy size for stress/strain. size does not matter. just write something that can be
      // extracted later
      CORE::LINALG::Matrix<1, 1> dummy(true);

      // write dummy stress
      {
        CORE::COMM::PackBuffer data;
        AddtoPack(data, dummy);
        data.StartPacking();
        AddtoPack(data, dummy);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }

      // write dummy strain
      {
        CORE::COMM::PackBuffer data;
        AddtoPack(data, dummy);
        data.StartPacking();
        AddtoPack(data, dummy);
        std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
      }
    }
    break;
    case calc_struct_constrvol:
    {
      // create communicator
      const Epetra_Comm& Comm = discretization.Comm();

      // We are not interested in volume of ghosted elements
      if (Comm.MyPID() == Owner())
      {
        // element geometry update
        Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
        if (disp == Teuchos::null) dserror("Cannot get state vector 'displacement'");
        std::vector<double> mydisp(lm.size());
        DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
        const int numdim = 3;
        CORE::LINALG::SerialDenseMatrix xscurr(NumNode(), numdim);  // material coord. of element
        SpatialConfiguration(xscurr, mydisp);
        // call submethod for volume evaluation and store rseult in third systemvector
        double volumeele = ComputeConstrVols(xscurr, NumNode());
        elevec3[0] = volumeele;
      }
    }
    break;
    case calc_struct_volconstrstiff:
    {
      // element geometry update
      Teuchos::RCP<const Epetra_Vector> disp = discretization.GetState("displacement");
      if (disp == Teuchos::null) dserror("Cannot get state vector 'displacement'");
      std::vector<double> mydisp(lm.size());
      DRT::UTILS::ExtractMyValues(*disp, mydisp, lm);
      const int numdim = 3;
      CORE::LINALG::SerialDenseMatrix xscurr(NumNode(), numdim);  // material coord. of element
      SpatialConfiguration(xscurr, mydisp);
      double volumeele;
      // first partial derivatives
      Teuchos::RCP<CORE::LINALG::SerialDenseVector> Vdiff1 =
          Teuchos::rcp(new CORE::LINALG::SerialDenseVector);
      // second partial derivatives
      Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> Vdiff2 =
          Teuchos::rcp(new CORE::LINALG::SerialDenseMatrix);

      // get projection method
      Teuchos::RCP<DRT::Condition> condition =
          params.get<Teuchos::RCP<DRT::Condition>>("condition");
      const std::string* projtype = condition->GetIf<std::string>("projection");

      if (projtype != nullptr)
      {
        // call submethod to compute volume and its derivatives w.r.t. to current displ.
        if (*projtype == "yz")
        {
          ComputeVolDeriv(xscurr, NumNode(), numdim * NumNode(), volumeele, Vdiff1, Vdiff2, 0, 0);
        }
        else if (*projtype == "xz")
        {
          ComputeVolDeriv(xscurr, NumNode(), numdim * NumNode(), volumeele, Vdiff1, Vdiff2, 1, 1);
        }
        else if (*projtype == "xy")
        {
          ComputeVolDeriv(xscurr, NumNode(), numdim * NumNode(), volumeele, Vdiff1, Vdiff2, 2, 2);
        }
        else
        {
          ComputeVolDeriv(xscurr, NumNode(), numdim * NumNode(), volumeele, Vdiff1, Vdiff2);
        }
      }
      else
        ComputeVolDeriv(xscurr, NumNode(), numdim * NumNode(), volumeele, Vdiff1, Vdiff2);

      // update rhs vector and corresponding column in "constraint" matrix
      elevec1 = *Vdiff1;
      elevec2 = *Vdiff1;
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
int DRT::ELEMENTS::Bele3::EvaluateNeumann(Teuchos::ParameterList& params,
    DRT::Discretization& discretization, DRT::Condition& condition, std::vector<int>& lm,
    CORE::LINALG::SerialDenseVector& elevec1, CORE::LINALG::SerialDenseMatrix* elemat1)
{
  return 0;
}


/*----------------------------------------------------------------------*
 * Compute Volume enclosed by surface.                          tk 10/07*
 * ---------------------------------------------------------------------*/
double DRT::ELEMENTS::Bele3::ComputeConstrVols(
    const CORE::LINALG::SerialDenseMatrix& xc, const int numnode)
{
  double V = 0.0;

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = 0; indc < 3; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    CORE::LINALG::SerialDenseMatrix ab = xc;
    CORE::LINALG::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const CORE::FE::IntegrationPoints2D intpoints(getOptimalGaussrule());
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    CORE::LINALG::SerialDenseVector funct(numnode);
    CORE::LINALG::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      CORE::FE::shape_function_2D(funct, e0, e1, Shape());
      CORE::FE::shape_function_2D_deriv1(deriv, e0, e1, Shape());

      double detA;
      // compute "metric tensor" deriv*ab, which is a 2x3 matrix with zero indc'th column
      CORE::LINALG::SerialDenseMatrix metrictensor(2, 3);
      CORE::LINALG::multiply(metrictensor, deriv, ab);
      // CORE::LINALG::SerialDenseMatrix metrictensor(2,2);
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
void DRT::ELEMENTS::Bele3::ComputeVolDeriv(const CORE::LINALG::SerialDenseMatrix& xc,
    const int numnode, const int ndof, double& V,
    Teuchos::RCP<CORE::LINALG::SerialDenseVector> Vdiff1,
    Teuchos::RCP<CORE::LINALG::SerialDenseMatrix> Vdiff2, const int minindex, const int maxindex)
{
  // necessary constants
  const int numdim = 3;
  const double invnumind = 1.0 / (maxindex - minindex + 1.0);

  // initialize
  V = 0.0;
  Vdiff1->size(ndof);
  if (Vdiff2 != Teuchos::null) Vdiff2->shape(ndof, ndof);

  // Volume is calculated by evaluating the integral
  // 1/3*int_A(x dydz + y dxdz + z dxdy)

  // we compute the three volumes separately
  for (int indc = minindex; indc < maxindex + 1; indc++)
  {
    // split current configuration between "ab" and "c"
    // where a!=b!=c and a,b,c are in {x,y,z}
    CORE::LINALG::SerialDenseMatrix ab = xc;
    CORE::LINALG::SerialDenseVector c(numnode);
    for (int i = 0; i < numnode; i++)
    {
      ab(i, indc) = 0.0;   // project by z_i = 0.0
      c(i) = xc(i, indc);  // extract z coordinate
    }
    // index of variables a and b
    int inda = (indc + 1) % 3;
    int indb = (indc + 2) % 3;

    // get gaussrule
    const CORE::FE::IntegrationPoints2D intpoints(getOptimalGaussrule());
    int ngp = intpoints.nquad;

    // allocate vector for shape functions and matrix for derivatives
    CORE::LINALG::SerialDenseVector funct(numnode);
    CORE::LINALG::SerialDenseMatrix deriv(2, numnode);

    /*----------------------------------------------------------------------*
     |               start loop over integration points                     |
     *----------------------------------------------------------------------*/
    for (int gpid = 0; gpid < ngp; ++gpid)
    {
      const double e0 = intpoints.qxg[gpid][0];
      const double e1 = intpoints.qxg[gpid][1];

      // get shape functions and derivatives of shape functions in the plane of the element
      CORE::FE::shape_function_2D(funct, e0, e1, Shape());
      CORE::FE::shape_function_2D_deriv1(deriv, e0, e1, Shape());

      // evaluate Jacobi determinant, for projected dA*
      std::vector<double> normal(numdim);
      double detA;
      // compute "metric tensor" deriv*xy, which is a 2x3 matrix with zero 3rd column
      CORE::LINALG::SerialDenseMatrix metrictensor(2, numdim);
      CORE::LINALG::multiply(metrictensor, deriv, ab);
      // metrictensor.Multiply('N','T',1.0,dxyzdrs,dxyzdrs,0.0);
      detA = metrictensor(0, inda) * metrictensor(1, indb) -
             metrictensor(0, indb) * metrictensor(1, inda);
      const double dotprodc = funct.dot(c);
      // add weighted volume at gausspoint
      V -= dotprodc * detA * intpoints.qwgt[gpid];

      //-------- compute first derivative
      for (int i = 0; i < numnode; i++)
      {
        (*Vdiff1)[3 * i + inda] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(0, i) * metrictensor(1, indb) - metrictensor(0, indb) * deriv(1, i));
        (*Vdiff1)[3 * i + indb] +=
            invnumind * intpoints.qwgt[gpid] * dotprodc *
            (deriv(1, i) * metrictensor(0, inda) - metrictensor(1, inda) * deriv(0, i));
        (*Vdiff1)[3 * i + indc] += invnumind * intpoints.qwgt[gpid] * funct[i] * detA;
      }

      //-------- compute second derivative
      if (Vdiff2 != Teuchos::null)
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

BACI_NAMESPACE_CLOSE
