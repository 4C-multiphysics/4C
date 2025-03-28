// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_browniandyn_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_spring.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_torsion3.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) cyron 08/08|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Torsion3::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
  {
    act = params_interface().get_action_type();
  }
  else  // Todo remove as soon as old structural time integration is gone
  {
    // get the action required
    std::string action = params.get<std::string>("action", "calc_none");
    if (action == "calc_none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_linstiff")
      act = Core::Elements::struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      act = Core::Elements::struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      act = Core::Elements::struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      act = Core::Elements::struct_calc_nlnstifflmass;
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
    else if (action == "calc_struct_update_istep")
      act = Core::Elements::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = Core::Elements::struct_calc_reset_istep;
    else if (action == "calc_struct_ptcstiff")
      act = Core::Elements::struct_calc_ptcstiff;
    else
    {
      std::cout << action << std::endl;
      FOUR_C_THROW("Unknown type of action for Torsion3");
    }
  }

  switch (act)
  {
    case Core::Elements::struct_calc_ptcstiff:
    {
      // nothing to do
    }
    break;
    /*in case that only linear stiffness matrix is required b3_nlstiffmass is called with zero
     displacement and residual values*/
    case Core::Elements::struct_calc_linstiff:
    {
      // only nonlinear case implemented!
      FOUR_C_THROW("linear stiffness matrix called, but not implemented");
    }
    break;
    // calculate internal energy
    case Core::Elements::struct_calc_energy:
    {
      // need current global displacement and get them from discretization
      // making use of the local-to-global map lm one can extract current displacemnet and residual
      // values for each degree of freedom
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      t3_energy(params, mydisp, &elevec1);
    }
    break;

    // nonlinear stiffness and mass matrix are calculated even if only nonlinear stiffness matrix is
    // required
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_internalforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacemnet and residual
      // values for each degree of freedom
      //
      // get element displcements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
      // get residual displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (res == nullptr) FOUR_C_THROW("Cannot get state vectors 'residual displacement'");
      std::vector<double> myres = Core::FE::extract_values(*res, lm);

      /*first displacement vector is modified for proper element evaluation in case of periodic
       *boundary conditions; in case that no periodic boundary conditions are to be applied the
       *following code line may be ignored or deleted*/
      // Todo method is deprecated. overhaul it if needed
      //      node_shift<3,3>(params,mydisp);


      // for engineering strains instead of total lagrange use t3_nlnstiffmass2
      if (act == Core::Elements::struct_calc_nlnstiffmass)
        t3_nlnstiffmass(mydisp, &elemat1, &elemat2, &elevec1);
      else if (act == Core::Elements::struct_calc_nlnstifflmass)
        t3_nlnstiffmass(mydisp, &elemat1, &elemat2, &elevec1);
      else if (act == Core::Elements::struct_calc_nlnstiff)
        t3_nlnstiffmass(mydisp, &elemat1, nullptr, &elevec1);
      else if (act == Core::Elements::struct_calc_internalforce)
        t3_nlnstiffmass(mydisp, nullptr, nullptr, &elevec1);


      /*
      //the following code block can be used to check quickly whether the nonlinear stiffness matrix
      is calculated
      //correctly or not by means of a numerically approximated stiffness matrix
      //The code block will work for all higher order elements.
      //if(Id() == 3) //limiting the following tests to certain element numbers
      {
        //assuming the same number of DOF for all nodes
        int numdof = NumDofPerNode(*(Nodes()[0]));
        int nnode  = num_node();

        //variable to store numerically approximated stiffness matrix
        Core::LinAlg::SerialDenseMatrix stiff_approx;
        stiff_approx.Shape(numdof*nnode,numdof*nnode);


        //relative error of numerically approximated stiffness matrix
        Core::LinAlg::SerialDenseMatrix stiff_relerr;
        stiff_relerr.Shape(numdof*nnode,numdof*nnode);

        //characteristic length for numerical approximation of stiffness
        double h_rel = 1e-6;

        //flag indicating whether approximation leads to significant relative error
        int outputflag = 0;

        //calculating strains in new configuration
        for(int i=0; i<numdof; i++) //for all dof
        {
          for(int k=0; k<nnode; k++)//for all nodes
          {

            Core::LinAlg::SerialDenseVector force_aux;
            force_aux.Size(numdof*nnode);

            //create new displacement and velocity vectors in order to store artificially modified
      displacements
            //vector<double> vel_aux(myvel);
            std::vector<double> disp_aux(mydisp);

            //modifying displacement artificially (for numerical derivative of internal forces):
            disp_aux[numdof*k + i] += h_rel;
            //vel_aux[numdof*k + i]  += h_rel / params.get<double>("delta time",0.01);

            t3_nlnstiffmass(disp_aux,nullptr,nullptr,&force_aux);

            //computing derivative d(fint)/du numerically by finite difference
            for(int u = 0 ; u < numdof*nnode ; u++ )
              stiff_approx(u,k*numdof+i) = ( pow(force_aux[u],2) - pow(elevec1(u),2) )/ (h_rel *
      (force_aux[u] + elevec1(u) ) );

          } //for(int k=0; k<nnode; k++)//for all nodes
        } //for(int i=0; i<numdof; i++) //for all dof


        for(int line=0; line<numdof*nnode; line++)
        {
          for(int col=0; col<numdof*nnode; col++)
          {
            stiff_relerr(line,col)= fabs( ( pow(elemat1(line,col),2) - pow(stiff_approx(line,col),2)
      )/ ( (elemat1(line,col) + stiff_approx(line,col)) * elemat1(line,col) ));

            //suppressing small entries whose effect is only confusing and NaN entries (which arise
      due to zero entries) if ( fabs( stiff_relerr(line,col) ) < h_rel*500 || isnan(
      stiff_relerr(line,col)) || elemat1(line,col) == 0) //isnan = is not a number
              stiff_relerr(line,col) = 0;

            if ( stiff_relerr(line,col) > 0)
              outputflag = 1;

          } //for(int col=0; col<numdof*nnode; col++)
        } //for(int line=0; line<numdof*nnode; line++)

        if(outputflag ==1)
        {
          std::cout<<"\n\n actually calculated stiffness matrix in Element "<<Id()<<": "<< elemat1;
          std::cout<<"\n\n approximated stiffness matrix in Element "<<Id()<<": "<< stiff_approx;
          std::cout<<"\n\n rel error stiffness matrix in Element "<<Id()<<": "<< stiff_relerr;
        }

      } //end of section in which numerical approximation for stiffness matrix is computed
      */
    }
    break;
    case Core::Elements::struct_calc_update_istep:
    case Core::Elements::struct_calc_reset_istep:
    case Core::Elements::struct_calc_stress:
      break;
    case Core::Elements::struct_calc_recover:
    {
      // do nothing here
      break;
    }

    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    default:
      std::cout << "\ncalled element with action type " << action_type_to_string(act);
      FOUR_C_THROW("Unknown type of action for Torsion3");
      break;
  }
  return 0;
}

/*-----------------------------------------------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public) cyron 03/08|
 *----------------------------------------------------------------------------------------------------------*/

int Discret::Elements::Torsion3::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  /*torsion spring assumed to be infinitesimally small and thus no Neumann boundary conditions
   * can be assigned to this element*/
  return 0;
}


/*--------------------------------------------------------------------------------------*
 | calculation of elastic energy                                             cyron 12/10|
 *--------------------------------------------------------------------------------------*/
void Discret::Elements::Torsion3::t3_energy(Teuchos::ParameterList& params,
    std::vector<double>& disp, Core::LinAlg::SerialDenseVector* intenergy)
{
  // current node position (first entries 0,1,2 for first node, 3,4,5 for second node , 6,7,8 for
  // third node)
  Core::LinAlg::Matrix<9, 1> xcurr;

  // current nodal position
  for (int j = 0; j < 3; ++j)
  {
    xcurr(j) = nodes()[0]->x()[j] + disp[j];          // first node
    xcurr(j + 3) = nodes()[1]->x()[j] + disp[3 + j];  // second node
    xcurr(j + 6) = nodes()[2]->x()[j] + disp[6 + j];  // third node
  }

  // auxiliary vector for both internal force and stiffness matrix
  Core::LinAlg::Matrix<6, 1> aux;
  for (int j = 0; j < 6; ++j) aux(j) = xcurr(j + 3) - xcurr(j);

  // current length of vectors 1-->2  and 2-->3
  Core::LinAlg::Matrix<2, 1> lcurr;
  for (int j = 0; j < 2; ++j)
    lcurr(j) = sqrt(pow(aux(3 * j), 2) + pow(aux(3 * j + 1), 2) + pow(aux(3 * j + 2), 2));

  // computing the change of angle theta (equation 3.3)
  double deltatheta = 0.0;
  double dotprod = 0.0;
  double s = 0.0;

  for (int j = 0; j < 3; ++j) dotprod += aux(j) * aux(3 + j);

  s = dotprod / lcurr(0) / lcurr(1);
  deltatheta = acos(s);

  // spring constant from material law
  std::shared_ptr<const Core::Mat::Material> currmat = material();
  double spring = 0.0;

  // assignment of material parameters; only spring material is accepted for this element
  switch (currmat->material_type())
  {
    case Core::Materials::m_spring:  // only elastic spring supported
    {
      const Mat::Spring* actmat = static_cast<const Mat::Spring*>(currmat.get());
      spring = actmat->stiffness();
    }
    break;
    default:
      FOUR_C_THROW("unknown or improper type of material law");
  }

  if (bendingpotential_ == quadratic)
    (*intenergy)(0) = 0.5 * spring * pow(deltatheta, 2);
  else if (bendingpotential_ == cosine)
    (*intenergy)(0) = spring * (1 - s);
  else
    FOUR_C_THROW(
        "\n No such bending potential. Possible bending potentials: \n quadratic \n cosine");


}  // t3_energy

/*--------------------------------------------------------------------------------------*
 | evaluate nonlinear stiffness matrix and internal forces                    cyron 03/10|
 *--------------------------------------------------------------------------------------*/
void Discret::Elements::Torsion3::t3_nlnstiffmass(std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force)
{
  // current node position (first entries 0,1,2 for first node, 3,4,5 for second node , 6,7,8 for
  // third node)
  Core::LinAlg::Matrix<9, 1> xcurr(Core::LinAlg::Initialization::zero);

  // current nodal position
  for (int j = 0; j < 3; ++j)
  {
    xcurr(j) = nodes()[0]->x()[j] + disp[j];          // first node
    xcurr(j + 3) = nodes()[1]->x()[j] + disp[3 + j];  // second node
    xcurr(j + 6) = nodes()[2]->x()[j] + disp[6 + j];  // third node
  }

  // auxiliary vector for both internal force and stiffness matrix
  Core::LinAlg::Matrix<6, 1> aux(Core::LinAlg::Initialization::zero);
  for (int j = 0; j < 6; ++j) aux(j) = xcurr(j + 3) - xcurr(j);

  // current length of vectors 1-->2  and 2-->3
  Core::LinAlg::Matrix<2, 1> lcurr(Core::LinAlg::Initialization::zero);
  for (int j = 0; j < 2; ++j)
    lcurr(j) = sqrt(pow(aux(3 * j), 2) + pow(aux(3 * j + 1), 2) + pow(aux(3 * j + 2), 2));

  // computing the change of angle theta (equation 3.3)
  double deltatheta = 0.0;
  double dotprod = 0.0;
  double s = 0.0;

  for (int j = 0; j < 3; ++j) dotprod += aux(j) * aux(3 + j);

  s = dotprod / lcurr(0) / lcurr(1);
  // Owing to round-off errors the variable s can be slightly
  // outside the admissible range [-1.0;1.0]. We take care for this
  // preventing potential floating point exceptions in acos(s)
  if (s > 1.0)
  {
    if ((s - 1.0) > 1.0e-14)
      FOUR_C_THROW("s out of admissible range [-1.0;1.0]");
    else  // tiny adaptation of s accounting for round-off errors
      s = 1.0;
  }
  if (s < -1.0)
  {
    if ((s + 1.0) < -1.0e-14)
      FOUR_C_THROW("s out of admissible range [-1.0;1.0]");
    else  // tiny adaptation of s accounting for round-off errors
      s = -1.0;
  }

  deltatheta = acos(s);


  // variation of theta (equation 3.4)
  Core::LinAlg::Matrix<9, 1> grtheta;

  for (int j = 0; j < 3; ++j)
  {  // virtual displacement of node 1 and 3
    grtheta(j) = -aux(3 + j) / lcurr(0) / lcurr(1) + dotprod * aux(j) / pow(lcurr(0), 3) / lcurr(1);
    grtheta(6 + j) =
        aux(j) / lcurr(0) / lcurr(1) - dotprod * aux(3 + j) / lcurr(0) / pow(lcurr(1), 3);
  }

  for (int j = 0; j < 3; ++j)  // virtual displacement of node 2
    grtheta(3 + j) = -grtheta(j) - grtheta(j + 6);

  // auxiliary matrix for stiffness matrix (equation 3.9 and equation 3.10)
  Core::LinAlg::Matrix<6, 3> A;
  Core::LinAlg::Matrix<6, 3> B;

  for (int j = 0; j < 3; ++j)
  {
    for (int i = 0; i < 3; ++i)
    {
      A(j, i) = aux(3 + j) * aux(i) / pow(lcurr(0), 3) / lcurr(1) +
                aux(3 + i) * aux(j) / pow(lcurr(0), 3) / lcurr(1) -
                3 * dotprod * aux(j) * aux(i) / pow(lcurr(0), 5) / lcurr(1);
      A(3 + j, i) = aux(j) * aux(3 + i) / lcurr(0) / pow(lcurr(1), 3) +
                    aux(i) * aux(3 + j) / lcurr(0) / pow(lcurr(1), 3) -
                    3 * dotprod * aux(3 + j) * aux(3 + i) / lcurr(0) / pow(lcurr(1), 5);
    }
    A(j, j) += dotprod / pow(lcurr(0), 3) / lcurr(1);
    A(3 + j, j) += dotprod / lcurr(0) / pow(lcurr(1), 3);
  }

  for (int j = 0; j < 3; ++j)
  {
    for (int i = 0; i < 3; ++i)
    {
      B(j, i) = -aux(j) * aux(i) / pow(lcurr(0), 3) / lcurr(1) -
                aux(3 + i) * aux(3 + j) / lcurr(0) / pow(lcurr(1), 3) +
                dotprod * aux(3 + j) * aux(i) / pow(lcurr(0), 3) / pow(lcurr(1), 3);
      B(3 + j, i) = -aux(3 + j) * aux(3 + i) / lcurr(0) / pow(lcurr(1), 3) -
                    aux(i) * aux(j) / pow(lcurr(0), 3) / lcurr(1) +
                    dotprod * aux(j) * aux(3 + i) / pow(lcurr(0), 3) / pow(lcurr(1), 3);
    }
    B(j, j) += 1 / lcurr(0) / lcurr(1);
    B(3 + j, j) += 1 / lcurr(0) / lcurr(1);
  }

  // spring constant from material law
  std::shared_ptr<const Core::Mat::Material> currmat = material();
  double spring = 0.0;

  // assignment of material parameters; only spring material is accepted for this element
  switch (currmat->material_type())
  {
    case Core::Materials::m_spring:  // only elastic spring supported
    {
      const Mat::Spring* actmat = static_cast<const Mat::Spring*>(currmat.get());
      spring = actmat->stiffness();
    }
    break;
    default:
      FOUR_C_THROW("unknown or improper type of material law");
  }

  // bending potential quadratic
  if (bendingpotential_ == quadratic)
  {
    if ((1 - s) <= 0.00000000001)
    {
      // calculation of the linear stiffness matrix (if the angle between the trusses is almost
      // zero)

      // internal forces (equation 3.17)
      if (force != nullptr)
      {
        for (int j = 0; j < 9; ++j) (*force)(j) = -spring * grtheta(j);
      }


      // stiffness matrix (equation 3.19)
      if (stiffmatrix != nullptr)
      {
        for (int j = 0; j < 3; ++j)
        {
          for (int i = 0; i < 3; ++i)
          {
            (*stiffmatrix)(j, i) = -spring * (-A(j, i));
            (*stiffmatrix)(j, 3 + i) = -spring * (A(j, i) + B(j + 3, i));
            (*stiffmatrix)(j, 6 + i) = -spring * (-B(j + 3, i));
            (*stiffmatrix)(3 + j, i) = -spring * (A(j, i) + B(j, i));
            (*stiffmatrix)(3 + j, 3 + i) =
                -spring * (-A(j, i) - A(j + 3, i) - B(j, i) - B(j + 3, i));
            (*stiffmatrix)(3 + j, 6 + i) = -spring * (A(j + 3, i) + B(j + 3, i));
            (*stiffmatrix)(6 + j, i) = -spring * (-B(j, i));
            (*stiffmatrix)(6 + j, 3 + i) = -spring * (A(j + 3, i) + B(j, i));
            (*stiffmatrix)(6 + j, 6 + i) = -spring * (-A(j + 3, i));
          }
        }
      }
    }  // if (linear stiffness matrix)

    else
    {
      // internal forces (equation 3.6) (if the angle between the trusses is NOT almost zero)
      if (force != nullptr)
      {
        for (int j = 0; j < 9; ++j)
          (*force)(j) = -1 / sqrt(1 - s * s) * deltatheta * spring * grtheta(j);
      }


      // stiffness matrix (equation 3.8)
      if (stiffmatrix != nullptr)
      {
        for (int i = 0; i < 9; ++i)
        {
          for (int j = 0; j < 9; ++j)
            (*stiffmatrix)(i, j) =
                1 / (1 - s * s) * spring * grtheta(i) * grtheta(j) -
                s / pow(sqrt(1 - s * s), 3) * deltatheta * spring * grtheta(i) * grtheta(j);
        }

        for (int j = 0; j < 3; ++j)  // equation 3.13
        {
          for (int i = 0; i < 3; ++i)
          {
            (*stiffmatrix)(j, i) += spring * deltatheta * (-1 / sqrt(1 - s * s)) * (-A(j, i));
            (*stiffmatrix)(j, 3 + i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (A(j, i) + B(j + 3, i));
            (*stiffmatrix)(j, 6 + i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (-B(j + 3, i));
            (*stiffmatrix)(3 + j, i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (A(j, i) + B(j, i));
            (*stiffmatrix)(3 + j, 3 + i) += spring * deltatheta * (-1 / sqrt(1 - s * s)) *
                                            (-A(j, i) - A(j + 3, i) - B(j, i) - B(j + 3, i));
            (*stiffmatrix)(3 + j, 6 + i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (A(j + 3, i) + B(j + 3, i));
            (*stiffmatrix)(6 + j, i) += spring * deltatheta * (-1 / sqrt(1 - s * s)) * (-B(j, i));
            (*stiffmatrix)(6 + j, 3 + i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (A(j + 3, i) + B(j, i));
            (*stiffmatrix)(6 + j, 6 + i) +=
                spring * deltatheta * (-1 / sqrt(1 - s * s)) * (-A(j + 3, i));
          }
        }
      }  // stiffness matrix

    }  // else (theta NOT almost zero)

  }  // bending potential quadratic

  else
    // bending potential cosine
    if (bendingpotential_ == cosine)
    {
      // internal forces (equation 3.16, same as for theta almost zero)
      if (force != nullptr)
      {
        for (int j = 0; j < 9; ++j) (*force)(j) = -spring * grtheta(j);
      }


      // stiffness matrix (equation 3.17, same as for theta almost zero)
      if (stiffmatrix != nullptr)
      {
        for (int j = 0; j < 3; ++j)
        {
          for (int i = 0; i < 3; ++i)
          {
            (*stiffmatrix)(j, i) = -spring * (-A(j, i));
            (*stiffmatrix)(j, 3 + i) = -spring * (A(j, i) + B(j + 3, i));
            (*stiffmatrix)(j, 6 + i) = -spring * (-B(j + 3, i));
            (*stiffmatrix)(3 + j, i) = -spring * (A(j, i) + B(j, i));
            (*stiffmatrix)(3 + j, 3 + i) =
                -spring * (-A(j, i) - A(j + 3, i) - B(j, i) - B(j + 3, i));
            (*stiffmatrix)(3 + j, 6 + i) = -spring * (A(j + 3, i) + B(j + 3, i));
            (*stiffmatrix)(6 + j, i) = -spring * (-B(j, i));
            (*stiffmatrix)(6 + j, 3 + i) = -spring * (A(j + 3, i) + B(j, i));
            (*stiffmatrix)(6 + j, 6 + i) = -spring * (-A(j + 3, i));
          }
        }
      }  // stiffness matrix
    }  // bending potential cosine

    else
      std::cout
          << "\n No such bending potential. Possible bending potentials: \n quadratic \n cosine"
          << std::endl;


}  // nlnstiffmass

/*-----------------------------------------------------------------------------------------------------------*
 | shifts nodes so that proper evaluation is possible even in case of periodic boundary conditions;
 if two   | | nodes within one element are separated by a periodic boundary, one of them is shifted
 such that the final | | distance in R^3 is the same as the initial distance in the periodic space;
 the shift affects computation  | | on element level within that very iteration step, only (no
 change in global variables performed)          |                                 | | (public) cyron
 10/09|
 *----------------------------------------------------------------------------------------------------------*/
template <int nnode, int ndim>  // number of nodes, number of dimensions
inline void Discret::Elements::Torsion3::node_shift(
    Teuchos::ParameterList& params,  //!< parameter list
    std::vector<double>& disp)       //!< element disp vector
{
  FOUR_C_THROW(
      "Torsion3::node_shift is deprecated; if needed adapt parameter handling according to "
      "parameter interface pointer first!");

}  // Discret::Elements::Torsion3::node_shift

FOUR_C_NAMESPACE_CLOSE
