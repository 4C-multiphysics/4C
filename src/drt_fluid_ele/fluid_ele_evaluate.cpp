/*!
\file fluid_ele_evaluate.cpp
\brief

<pre>
Maintainer: Volker Gravemeier & Andreas Ehrl
            {vgravem,ehrl}@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089-289-15245/15252
</pre>
*/


#include "fluid_ele_factory.H"

#include "fluid_ele.H"
#include "fluid_ele_utils.H"
#include "fluid_genalpha_resVMM.H"
#include "fluid_ele_interface.H"
#include "fluid_ele_parameter.H"

#include "../drt_fem_general/drt_utils_nurbs_shapefunctions.H"

#include "../drt_geometry/position_array.H"

#include "../drt_inpar/inpar_fluid.H"

#include "../drt_lib/drt_utils.H"

#include "../drt_mat/arrhenius_pv.H"
#include "../drt_mat/carreauyasuda.H"
#include "../drt_mat/ferech_pv.H"
#include "../drt_mat/mixfrac.H"
#include "../drt_mat/modpowerlaw.H"
#include "../drt_mat/newtonianfluid.H"
#include "../drt_mat/permeablefluid.H"
#include "../drt_mat/sutherland.H"
#include "../drt_mat/yoghurt.H"

#include "../drt_nurbs_discret/drt_nurbs_discret.H"

#include "../drt_opti/topopt_fluidAdjoint3_interface.H"
#include "../drt_opti/topopt_fluidAdjoint3_impl_parameter.H"

//#include "../drt_inpar/inpar_turbulence.H"
// include define flags for turbulence models under development
//#include "../drt_fluid/fluid_turbulence_defines.H"

using namespace DRT::UTILS;

/*
  Depending on the type of action and the element type (tet, hex etc.),
  the elements allocate common static arrays.

  */


/*---------------------------------------------------------------------*
|  converts a string into an action for this element                   |
*----------------------------------------------------------------------*/
DRT::ELEMENTS::Fluid::ActionType DRT::ELEMENTS::Fluid::convertStringToActionType(
              const string& action) const
{
  dsassert(action != "none", "No action supplied");

  DRT::ELEMENTS::Fluid::ActionType act = Fluid::none;
  if (action == "calc_fluid_systemmat_and_residual")
    act = Fluid::calc_fluid_systemmat_and_residual;
  else if (action == "calc_porousflow_fluid_coupling")
    act = Fluid::calc_porousflow_fluid_coupling;
  else if (action == "calc_loma_mono_odblock")
    act = Fluid::calc_loma_mono_odblock;
  else if (action == "calc_fluid_genalpha_sysmat_and_residual")
    act = Fluid::calc_fluid_genalpha_sysmat_and_residual;
  else if (action == "time update for subscales")
    act = Fluid::calc_fluid_genalpha_update_for_subscales;
  else if (action == "time average for subscales and residual")
    act = Fluid::calc_fluid_genalpha_average_for_subscales_and_residual;
  else if (action == "calc_dissipation")
    act = Fluid::calc_dissipation;
  else if (action == "calc model parameter multifractal subgid scales")
    act = Fluid::calc_model_params_mfsubgr_scales;
  else if (action == "calc_fluid_error")
    act = Fluid::calc_fluid_error;
  else if (action == "calc_turbulence_statistics")
    act = Fluid::calc_turbulence_statistics;
  else if (action == "calc_loma_statistics")
    act = Fluid::calc_loma_statistics;
  else if (action == "calc_turbscatra_statistics")
    act = Fluid::calc_turbscatra_statistics;
  else if (action == "calc_fluid_box_filter")
    act = Fluid::calc_fluid_box_filter;
  else if (action == "calc_smagorinsky_const")
    act = Fluid::calc_smagorinsky_const;
  else if (action == "get_gas_constant")
    act = Fluid::get_gas_constant;
  else if (action == "calc_node_normal")
    act = Fluid::calc_node_normal;
  else if (action == "integrate_shape")
    act = Fluid::integrate_shape;
  else if (action == "calc_divop")
    act = Fluid::calc_divop;
  else if (action == "calc_fluid_elementvolume")
    act = Fluid::calc_fluid_elementvolume;
  else if (action == "set_general_fluid_parameter")
    act = Fluid::set_general_fluid_parameter;
  else if (action == "set_time_parameter")
    act = Fluid::set_time_parameter;
  else if (action == "set_turbulence_parameter")
    act = Fluid::set_turbulence_parameter;
  else if (action == "set_loma_parameter")
    act = Fluid::set_loma_parameter;
  else if (action == "set_general_adjoint_parameter")
    act = Fluid::set_general_adjoint_parameter;
  else if (action == "set_adjoint_time_parameter")
    act = Fluid::set_adjoint_time_parameter;
  else if (action == "calc_adjoint_systemmat_and_residual")
    act = Fluid::calc_adjoint_systemmat_and_residual;
  else if (action == "AdjointNeumannBoundaryCondition")
    act = Fluid::calc_adjoint_neumann;
  else
  dserror("(%s) Unknown type of action for Fluid",action.c_str());
  return act;
}

/*---------------------------------------------------------------------*
|  Call the element to set all basic parameter                         |
*----------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidType::PreEvaluate(DRT::Discretization&                  dis,
                                            Teuchos::ParameterList&               p,
                                            Teuchos::RCP<LINALG::SparseOperator>  systemmatrix1,
                                            Teuchos::RCP<LINALG::SparseOperator>  systemmatrix2,
                                            Teuchos::RCP<Epetra_Vector>           systemvector1,
                                            Teuchos::RCP<Epetra_Vector>           systemvector2,
                                            Teuchos::RCP<Epetra_Vector>           systemvector3)
{
  const string action = p.get<string>("action","none");

  if (action == "set_general_fluid_parameter")
  {
    DRT::ELEMENTS::FluidEleParameter* fldpara = DRT::ELEMENTS::FluidEleParameter::Instance();
    fldpara->SetElementGeneralFluidParameter(p,dis.Comm().MyPID());
  }
  else if (action == "set_time_parameter")
  {
    DRT::ELEMENTS::FluidEleParameter* fldpara = DRT::ELEMENTS::FluidEleParameter::Instance();
    fldpara->SetElementTimeParameter(p);
  }
  else if (action == "set_turbulence_parameter")
  {
    DRT::ELEMENTS::FluidEleParameter* fldpara = DRT::ELEMENTS::FluidEleParameter::Instance();
    fldpara->SetElementTurbulenceParameter(p);
  }
  else if (action == "set_loma_parameter")
  {
    DRT::ELEMENTS::FluidEleParameter* fldpara = DRT::ELEMENTS::FluidEleParameter::Instance();
    fldpara->SetElementLomaParameter(p);
  }
  else if (action == "set_general_adjoint_parameter")
  {
    Teuchos::RCP<DRT::ELEMENTS::FluidAdjoint3ImplParameter> fldpara = DRT::ELEMENTS::FluidAdjoint3ImplParameter::Instance();
    fldpara->SetElementGeneralAdjointParameter(p);
  }
  else if (action == "set_adjoint_time_parameter")
  {
    Teuchos::RCP<DRT::ELEMENTS::FluidAdjoint3ImplParameter> fldpara = DRT::ELEMENTS::FluidAdjoint3ImplParameter::Instance();
    fldpara->SetElementAdjointTimeParameter(p);
  }

  return;
}


 /*----------------------------------------------------------------------*
 |  evaluate the element (public)                            g.bau 03/07|
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Fluid::Evaluate(Teuchos::ParameterList&            params,
                                    DRT::Discretization&      discretization,
                                    vector<int>&              lm,
                                    Epetra_SerialDenseMatrix& elemat1,
                                    Epetra_SerialDenseMatrix& elemat2,
                                    Epetra_SerialDenseVector& elevec1,
                                    Epetra_SerialDenseVector& elevec2,
                                    Epetra_SerialDenseVector& elevec3)
{
  // get the action required
  const string action = params.get<string>("action","none");
  const DRT::ELEMENTS::Fluid::ActionType act = convertStringToActionType(action);

  // get material
  RCP<MAT::Material> mat = Material();

  // get space dimensions
  const int nsd = getDimension(Shape());

  switch(act)
  {
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case calc_fluid_systemmat_and_residual:
    {
      switch(params.get<int>("physical type",INPAR::FLUID::incompressible))
      {
      case INPAR::FLUID::loma:
      {
          return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "loma")->Evaluate(
              this,
              discretization,
              lm,
              params,
              mat,
              elemat1,
              elemat2,
              elevec1,
              elevec2,
              elevec3 );
          break;
      }
      case INPAR::FLUID::poro:
      {
        return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "poro")->Evaluate(
            this,
            discretization,
            lm,
            params,
            mat,
            elemat1,
            elemat2,
            elevec1,
            elevec2,
            elevec3 );
        break;
      }
      default:
        return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "std")->Evaluate(
            this,
            discretization,
            lm,
            params,
            mat,
            elemat1,
            elemat2,
            elevec1,
            elevec2,
            elevec3);
      }

    }
    break;
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    // for the particular case of porous flow
    //-----------------------------------------------------------------------
    /***********************************************/
    case calc_porousflow_fluid_coupling:
    {
      if( mat->MaterialType() == INPAR::MAT::m_fluidporo)
      {
        return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "poro")->Evaluate(
            this,
            discretization,
            lm,
            params,
            mat,
            elemat1,
            elemat2,
            elevec1,
            elevec2,
            elevec3,
            true);
      }
      else
        dserror("Unknown material type for poroelasticity\n");
    }
    break;
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    // for evaluation of off-diagonal matrix block for monolithic
    // low-Mach-number solver
    //-----------------------------------------------------------------------
    case calc_loma_mono_odblock:
    {
      return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "loma")->Evaluate(
          this,
          discretization,
          lm,
          params,
          mat,
          elemat1,
          elemat2,
          elevec1,
          elevec2,
          elevec3,
          true);
    }
    break;
    //--------------------------------------------------
    // previous generalized-alpha (n+1) implementation
    //--------------------------------------------------
    case calc_fluid_genalpha_sysmat_and_residual:
    {
      return DRT::ELEMENTS::FluidGenalphaResVMMInterface::Impl(this)->Evaluate(
          this,
          params,
          discretization,
          lm,
          elemat1,
          elemat2,
          elevec1,
          elevec2,
          elevec3,
          mat);
    }
    break;
    case calc_fluid_error:
    {
      // integrate shape function for this element
      // (results assembled into element vector)
      // return DRT::ELEMENTS::FluidImplInterface::Impl(Shape(),"test")->ComputeError(this, params, mat, discretization, lm, elevec1);
      return  DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "std")->ComputeError(this, params, mat, discretization, lm, elevec1);
    }
    break;
    case calc_turbulence_statistics:
    {
      if (nsd == 3)
      {
        // do nothing if you do not own this element
        if(this->Owner() == discretization.Comm().MyPID())
        {
          // --------------------------------------------------
          // extract velocity and pressure from global
          // distributed vectors
          // --------------------------------------------------
          // velocity and pressure values (n+1)
          RCP<const Epetra_Vector> velnp
          = discretization.GetState("u and p (n+1,converged)");
          if (velnp==null) dserror("Cannot get state vector 'velnp'");

          // extract local values from the global vectors
          vector<double> mysol  (lm.size());
          DRT::UTILS::ExtractMyValues(*velnp,mysol,lm);

          vector<double> mydisp(lm.size());
          if(is_ale_)
          {
            // get most recent displacements
            RCP<const Epetra_Vector> dispnp
            =
                discretization.GetState("dispnp");

            if (dispnp==null)
            {
              dserror("Cannot get state vectors 'dispnp'");
            }

            DRT::UTILS::ExtractMyValues(*dispnp,mydisp,lm);
          }

          // integrate mean values
          const DiscretizationType distype = this->Shape();

          switch (distype)
          {
          case DRT::Element::hex8:
          {
            f3_calc_means<8>(discretization,mysol,mydisp,params);
            break;
          }
          case DRT::Element::hex20:
          {
            f3_calc_means<20>(discretization,mysol,mydisp,params);
            break;
          }
          case DRT::Element::hex27:
          {
            f3_calc_means<27>(discretization,mysol,mydisp,params);
            break;
          }
          case DRT::Element::nurbs8:
          {
            f3_calc_means<8>(discretization,mysol,mydisp,params);
            break;
          }
          case DRT::Element::nurbs27:
          {
            f3_calc_means<27>(discretization,mysol,mydisp,params);
            break;
          }
          default:
          {
            dserror("Unknown element type for mean value evaluation\n");
          }
          }
        }
      } // end if (nsd == 3)
      else dserror("action 'calc_turbulence_statistics' is a 3D specific action");
    }
    break;
    case calc_turbscatra_statistics:
    {
      if(nsd == 3)
      {
        // do nothing if you do not own this element
        if(this->Owner() == discretization.Comm().MyPID())
        {
          // --------------------------------------------------
          // extract velocity, pressure, and scalar from global
          // distributed vectors
          // --------------------------------------------------
          // velocity/pressure and scalar values (n+1)
          RCP<const Epetra_Vector> velnp
            = discretization.GetState("u and p (n+1,converged)");
          RCP<const Epetra_Vector> scanp
            = discretization.GetState("scalar (n+1,converged)");
          if (velnp==null || scanp==null)
            dserror("Cannot get state vectors 'velnp' and/or 'scanp'");

          // extract local values from the global vectors
          vector<double> myvelpre(lm.size());
          vector<double> mysca(lm.size());
          DRT::UTILS::ExtractMyValues(*velnp,myvelpre,lm);
          DRT::UTILS::ExtractMyValues(*scanp,mysca,lm);

          // integrate mean values
          const DiscretizationType distype = this->Shape();

          switch (distype)
          {
          case DRT::Element::hex8:
          {
            f3_calc_scatra_means<8>(discretization,myvelpre,mysca,params);
            break;
          }
          case DRT::Element::hex20:
          {
            f3_calc_scatra_means<20>(discretization,myvelpre,mysca,params);
            break;
          }
          case DRT::Element::hex27:
          {
            f3_calc_scatra_means<27>(discretization,myvelpre,mysca,params);
            break;
          }
          default:
          {
            dserror("Unknown element type for turbulent passive scalar mean value evaluation\n");
          }
          }
        }
      } // end if (nsd == 3)
      else dserror("action 'calc_loma_statistics' is a 3D specific action");
    }
    break;
    case calc_loma_statistics:
    {
      if(nsd == 3)
      {
        // do nothing if you do not own this element
        if(this->Owner() == discretization.Comm().MyPID())
        {
          // --------------------------------------------------
          // extract velocity, pressure, and temperature from
          // global distributed vectors
          // --------------------------------------------------
          // velocity/pressure and scalar values (n+1)
          RCP<const Epetra_Vector> velnp
          = discretization.GetState("u and p (n+1,converged)");
          RCP<const Epetra_Vector> scanp
          = discretization.GetState("scalar (n+1,converged)");
          if (velnp==null || scanp==null)
            dserror("Cannot get state vectors 'velnp' and/or 'scanp'");

          // extract local values from global vectors
          vector<double> myvelpre(lm.size());
          vector<double> mysca(lm.size());
          DRT::UTILS::ExtractMyValues(*velnp,myvelpre,lm);
          DRT::UTILS::ExtractMyValues(*scanp,mysca,lm);

          // get factor for equation of state
          const double eosfac = params.get<double>("eos factor",100000.0/287.0);

          // integrate mean values
          const DiscretizationType distype = this->Shape();

          switch (distype)
          {
          case DRT::Element::hex8:
          {
            f3_calc_loma_means<8>(discretization,myvelpre,mysca,params,eosfac);
            break;
          }
          case DRT::Element::hex20:
          {
            f3_calc_loma_means<20>(discretization,myvelpre,mysca,params,eosfac);
            break;
          }
          case DRT::Element::hex27:
          {
            f3_calc_loma_means<27>(discretization,myvelpre,mysca,params,eosfac);
            break;
          }
          default:
          {
            dserror("Unknown element type for low-Mach-number mean value evaluation\n");
          }
          }
        }
      } // end if (nsd == 3)
      else dserror("action 'calc_loma_statistics' is a 3D specific action");
    }
    break;
    case calc_fluid_box_filter:
    {
      if (nsd == 3)
      {
        const bool dyn_smagorinsky = params.get<bool>("LESmodel");

        // --------------------------------------------------
        // extract velocity and pressure from global
        // distributed vectors
        // --------------------------------------------------
        // velocity and pressure values (most recent
        // intermediate solution, i.e. n+alphaF for genalpha
        // and n+1 for one-step-theta)
        RCP<const Epetra_Vector> vel =
            discretization.GetState("u and p (trial)");

        if (vel==null)
        {
          dserror("Cannot get state vectors 'vel'");
        }

        // extract local values from the global vectors
        vector<double> myvel(lm.size());
        DRT::UTILS::ExtractMyValues(*vel,myvel,lm);

        // initialise the contribution of this element to the patch volume to zero
        double volume_contribution = 0;

        // integrate the convolution with the box filter function for this element
        // the results are assembled onto the *_hat arrays

        const DiscretizationType distype = this->Shape();
        switch (distype)
        {
        case DRT::Element::hex8:
        {
          this->f3_apply_box_filter<8>(dyn_smagorinsky,
              myvel,
              elevec1.Values(),
              elemat1.A(),
              elemat2.A(),
              volume_contribution);
          break;
        }
        case DRT::Element::tet4:
        {
          this->f3_apply_box_filter<4>(dyn_smagorinsky,
              myvel,
              elevec1.Values(),
              elemat1.A(),
              elemat2.A(),
              volume_contribution);
          break;
        }
        default:
        {
          dserror("Unknown element type for box filter application\n");
        }
        }

        // hand down the volume contribution to the time integration algorithm
        params.set<double>("volume_contribution",volume_contribution);
      } // end if (nsd == 3)
      else dserror("action 'calc_fluid_box_filter' is 3D specific action");
    }
    break;
    case calc_smagorinsky_const:
    {
      if(nsd == 3)
      {
        RCP<Epetra_MultiVector> filtered_vel                        =
            params.get<RCP<Epetra_MultiVector> >("col_filtered_vel");
        RCP<Epetra_MultiVector> col_filtered_reynoldsstress         =
            params.get<RCP<Epetra_MultiVector> >("col_filtered_reynoldsstress");
        RCP<Epetra_MultiVector> col_filtered_modeled_subgrid_stress =
            params.get<RCP<Epetra_MultiVector> >("col_filtered_modeled_subgrid_stress");

        double LijMij   = 0.0;
        double MijMij   = 0.0;
        double xcenter  = 0.0;
        double ycenter  = 0.0;
        double zcenter  = 0.0;

        const DiscretizationType distype = this->Shape();
        switch (distype)
        {
        case DRT::Element::hex8:
        {
          this->f3_calc_smag_const_LijMij_and_MijMij<8>(
              filtered_vel                       ,
              col_filtered_reynoldsstress        ,
              col_filtered_modeled_subgrid_stress,
              LijMij                             ,
              MijMij                             ,
              xcenter                            ,
              ycenter                            ,
              zcenter                            );
          break;
        }
        case DRT::Element::tet4:
        {
          this->f3_calc_smag_const_LijMij_and_MijMij<4>(
              filtered_vel                       ,
              col_filtered_reynoldsstress        ,
              col_filtered_modeled_subgrid_stress,
              LijMij                             ,
              MijMij                             ,
              xcenter                            ,
              ycenter                            ,
              zcenter                            );
          break;
        }
        default:
        {
          dserror("Unknown element type for box filter application\n");
        }
        }

        // set Cs_delta_sq without averaging (only clipping)
        if (abs(MijMij) < 1E-16) Cs_delta_sq_= 0.0;
        else  Cs_delta_sq_ = 0.5 * LijMij / MijMij;
        if (Cs_delta_sq_<0.0)
        {
          Cs_delta_sq_= 0.0;
        }

        params.set<double>("LijMij",LijMij);
        params.set<double>("MijMij",MijMij);
        params.set<double>("xcenter",xcenter);
        params.set<double>("ycenter",ycenter);
        params.set<double>("zcenter",zcenter);
      } // end if(nsd == 3)
      else dserror("action 'calc_smagorinsky_const' is a 3D specific action");
    }
    break;
    case calc_fluid_genalpha_update_for_subscales:
    {
      // the old subgrid-scale acceleration for the next timestep is calculated
      // on the fly, not stored on the element
      /*
                       ~n+1   ~n
               ~ n+1     u    - u     ~ n   / 1.0-gamma \
              acc    =   --------- - acc * |  ---------  |
                         gamma*dt           \   gamma   /

               ~ n       ~ n+1   / 1.0-gamma \
              acc    =    acc * |  ---------  |
       */

      const double dt     = params.get<double>("dt");
      const double gamma  = params.get<double>("gamma");

      // variable in space dimensions
      for(int rr=0;rr<nsd;++rr)
      {
        for(int mm=0;mm<svelnp_.N();++mm)
        {
          saccn_(rr,mm) =
              (svelnp_(rr,mm)-sveln_(rr,mm))/(gamma*dt)
              -
              saccn_(rr,mm)*(1.0-gamma)/gamma;
        }
      }

      // most recent subgrid-scale velocity becomes the old subscale velocity
      // for the next timestep
      //
      //  ~n   ~n+1
      //  u <- u
      //
      // variable in space dimensions
      for(int rr=0;rr<nsd;++rr)
      {
        for(int mm=0;mm<svelnp_.N();++mm)
        {
          sveln_(rr,mm)=svelnp_(rr,mm);
        }
      }
    }
    break;
    case calc_fluid_genalpha_average_for_subscales_and_residual:
    {
      if (nsd == 3)
      {
        if(this->Owner() == discretization.Comm().MyPID())
        {

          return DRT::ELEMENTS::FluidGenalphaResVMMInterface::Impl(this)->CalcResAvgs(
              this,
              params,
              discretization,
              lm,
              mat);
        }
      }
      else dserror("%i D elements does not support any averaging for subscales and residuals", nsd);
    }
    break;
    case calc_dissipation:
    {
      if (nsd == 3)
      {
        if (this->Owner() == discretization.Comm().MyPID()) // don't store values of gosted elements
        {
          return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "std")->CalcDissipation(
              this,
              params,
              discretization,
              lm,
              mat);
        }
      }
      else dserror("%i D elements does not support calculation of dissipation", nsd);
    }
    break;
    case calc_model_params_mfsubgr_scales:
    {
      if (nsd == 3)
      {
        // velocity values
        RCP<const Epetra_Vector> velnp = discretization.GetState("velnp");
        // fine-scale velocity values
        RCP<const Epetra_Vector> fsvelnp = discretization.GetState("fsvelnp");
        if (velnp==null or fsvelnp==null)
        {
          dserror("Cannot get state vectors");
        }

        // extract local values from the global vectors
        vector<double> myvel(lm.size());
        DRT::UTILS::ExtractMyValues(*velnp,myvel,lm);
        vector<double> myfsvel(lm.size());
        DRT::UTILS::ExtractMyValues(*fsvelnp,myfsvel,lm);

        const DiscretizationType distype = this->Shape();
        switch (distype)
        {
        case DRT::Element::hex8:
        {
          // don't store values of ghosted elements
          if (this->Owner() == discretization.Comm().MyPID())
          {
            this->f3_get_mf_params<8,3,DRT::Element::hex8>(params,mat,myvel,myfsvel);
          }
          break;
        }
        default:
        {
          dserror("Unknown element type for box filter application\n");
        }
        }
      }
      else dserror("%i D elements does not support calculation of model parameters", nsd);
    }
    case get_gas_constant:
    {
      if (mat->MaterialType()== INPAR::MAT::m_sutherland)
      {
        MAT::Sutherland* actmat = static_cast<MAT::Sutherland*>(mat.get());
        params.set("gas constant", actmat->GasConst());
      }
    }
    break;
    case calc_node_normal:
    {
      if (nsd == 3)
      {
        const DiscretizationType distype = this->Shape();
        switch (distype)
        {
        case DRT::Element::hex27:
        {
          this->ElementNodeNormal<DRT::Element::hex27>(params,discretization,lm,elevec1);
          break;
        }
        case DRT::Element::hex20:
        {
          this->ElementNodeNormal<DRT::Element::hex20>(params,discretization,lm,elevec1);
          break;
        }
        case DRT::Element::hex8:
        {
          this->ElementNodeNormal<DRT::Element::hex8>(params,discretization,lm,elevec1);
          break;
        }
        case DRT::Element::tet4:
        {
          this->ElementNodeNormal<DRT::Element::tet4>(params,discretization,lm,elevec1);
          break;
        }
        case DRT::Element::tet10:
        {
          this->ElementNodeNormal<DRT::Element::tet10>(params,discretization,lm,elevec1);
          break;
        }
        default:
        {
          dserror("Unknown element type for shape function integration\n");
        }
        }
      }
      else dserror("action 'calculate node normal' should also work in 2D, but 2D elements are not"
          " added to the template yet. Also it is not tested");
      break;
    }
    case integrate_shape:
    {
      // integrate shape function for this element
      // (results assembled into element vector)
      return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "std")->IntegrateShapeFunction(this, discretization, lm, elevec1);
    }
    case calc_divop:
    {
      // calculate the integrated divergence oprator
      return DRT::ELEMENTS::FluidFactory::ProvideImpl(Shape(), "std")->CalcDivOp(this, discretization, lm, elevec1);
    }
    case set_general_fluid_parameter:
    case set_time_parameter:
    case set_turbulence_parameter:
    case set_loma_parameter:
    case set_general_adjoint_parameter:
    case set_adjoint_time_parameter:
    case calc_adjoint_neumann: // this is done by the surface elements
      break;

    //-----------------------------------------------------------------------
    // adjoint implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case calc_adjoint_systemmat_and_residual:
    {
      return DRT::ELEMENTS::FluidAdjoint3ImplInterface::Impl(Shape())->Evaluate(
          this,
          discretization,
          lm,
          params,
          mat,
          elemat1,
          elevec1 );
      break;
    }
    default:
      dserror("Unknown type of action for Fluid");
  } // end of switch(act)

  return 0;
} // end of DRT::ELEMENTS::Fluid::Evaluate


/*----------------------------------------------------------------------*
 |  do nothing (public)                                      gammi 04/07|
 |                                                                      |
 |  The function is just a dummy. For fluid elements, the integration   |
 |  integration of volume Neumann conditions (body forces) takes place  |
 |  in the element. We need it there for the stabilisation terms!       |
 *----------------------------------------------------------------------*/
int DRT::ELEMENTS::Fluid::EvaluateNeumann(Teuchos::ParameterList&    params,
                                           DRT::Discretization&      discretization,
                                           DRT::Condition&           condition,
                                           vector<int>&              lm,
                                           Epetra_SerialDenseVector& elevec1,
                                           Epetra_SerialDenseMatrix* elemat1)
{
  return 0;
}

/*---------------------------------------------------------------------*
 | Calculate spatial mean values for channel flow          gammi 07/07 |
 *---------------------------------------------------------------------*/
template<int iel>
void DRT::ELEMENTS::Fluid::f3_calc_means(DRT::Discretization&     discretization,
                                          vector<double>&         solution,
                                          vector<double>&         displacement,
                                          Teuchos::ParameterList& params)
{
  // get view of solution and subgrid-viscosity vector
  LINALG::Matrix<4*iel,1> sol(&(solution[0]),true);

  // set element data
  const DiscretizationType distype = this->Shape();

  // the plane normal tells you in which plane the integration takes place
  const int normdirect = params.get<int>("normal direction to homogeneous plane");

  // the vector planes contains the coordinates of the homogeneous planes (in
  // wall normal direction)
  RCP<vector<double> > planes = params.get<RCP<vector<double> > >("coordinate vector for hom. planes");

  // get the pointers to the solution vectors
  RCP<vector<double> > sumarea= params.get<RCP<vector<double> > >("element layer area");

  RCP<vector<double> > sumu   = params.get<RCP<vector<double> > >("mean velocity u");
  RCP<vector<double> > sumv   = params.get<RCP<vector<double> > >("mean velocity v");
  RCP<vector<double> > sumw   = params.get<RCP<vector<double> > >("mean velocity w");
  RCP<vector<double> > sump   = params.get<RCP<vector<double> > >("mean pressure p");

  RCP<vector<double> > sumsqu = params.get<RCP<vector<double> > >("mean value u^2");
  RCP<vector<double> > sumsqv = params.get<RCP<vector<double> > >("mean value v^2");
  RCP<vector<double> > sumsqw = params.get<RCP<vector<double> > >("mean value w^2");
  RCP<vector<double> > sumuv  = params.get<RCP<vector<double> > >("mean value uv");
  RCP<vector<double> > sumuw  = params.get<RCP<vector<double> > >("mean value uw");
  RCP<vector<double> > sumvw  = params.get<RCP<vector<double> > >("mean value vw");
  RCP<vector<double> > sumsqp = params.get<RCP<vector<double> > >("mean value p^2");

  // get node coordinates of element
  LINALG::Matrix<3,iel>  xyze;
  DRT::Node** nodes = Nodes();
  for(int inode=0;inode<iel;inode++)
  {
    const double* x = nodes[inode]->X();
    xyze(0,inode)=x[0];
    xyze(1,inode)=x[1];
    xyze(2,inode)=x[2];
  }

  if(is_ale_)
  {
    for (int inode=0; inode<iel; inode++)
    {
      const int finode = 4*inode;

      xyze(0,inode) += displacement[ +finode];
      xyze(1,inode) += displacement[1+finode];
      xyze(2,inode) += displacement[2+finode];

      if(abs(displacement[normdirect+finode])>1e-6)
      {
        dserror("no sampling possible if homogeneous planes are not conserved\n");
      }
    }
  }

  if(distype == DRT::Element::hex8
     ||
     distype == DRT::Element::hex27
     ||
     distype == DRT::Element::hex20)
  {
    double min = xyze(normdirect,0);
    double max = xyze(normdirect,0);

    // set maximum and minimum value in wall normal direction
    for(int inode=0;inode<iel;inode++)
    {
      if(min > xyze(normdirect,inode))
      {
        min=xyze(normdirect,inode);
      }
      if(max < xyze(normdirect,inode))
      {
         max=xyze(normdirect,inode);
      }
    }

    // determine the ids of the homogeneous planes intersecting this element
    set<int> planesinele;
    for(unsigned nplane=0;nplane<planes->size();++nplane)
    {
    // get all available wall normal coordinates
      for(int nn=0;nn<iel;++nn)
      {
        if (min-2e-9 < (*planes)[nplane] && max+2e-9 > (*planes)[nplane])
        {
          planesinele.insert(nplane);
         }
      }
    }

    // remove lowest layer from planesinele to avoid double calculations. This is not done
    // for the first level (index 0) --- if deleted, shift the first integration point in
    // wall normal direction
    // the shift depends on the number of sampling planes in the element
    double shift=0;

    // set the number of planes which cut the element
    const int numplanesinele = planesinele.size();

    if(*planesinele.begin() != 0)
    {
      // this is not an element of the lowest element layer
      planesinele.erase(planesinele.begin());

      shift=2.0/(static_cast<double>(numplanesinele-1));
    }
    else
    {
      // this is an element of the lowest element layer. Increase the counter
      // in order to compute the total number of elements in one layer
      int* count = params.get<int*>("count processed elements");

      (*count)++;
    }

    // determine the orientation of the rst system compared to the xyz system
    int elenormdirect=-1;
    bool upsidedown =false;
    // the only thing of interest is how normdirect is oriented in the
    // element coordinate system
    if(xyze(normdirect,4)-xyze(normdirect,0)>2e-9)
    {
      // t aligned
      elenormdirect =2;
      cout << "upsidedown false" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)>2e-9)
    {
      // s aligned
      elenormdirect =1;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)>2e-9)
    {
      // r aligned
      elenormdirect =0;
    }
    else if(xyze(normdirect,4)-xyze(normdirect,0)<-2e-9)
    {
      cout << xyze(normdirect,4)-xyze(normdirect,0) << &endl;
      // -t aligned
      elenormdirect =2;
      upsidedown =true;
      cout << "upsidedown true" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)<-2e-9)
    {
      // -s aligned
      elenormdirect =1;
      upsidedown =true;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)<-2e-9)
    {
      // -r aligned
      elenormdirect =0;
      upsidedown =true;
    }
    else
    {
      dserror("cannot determine orientation of plane normal in local coordinate system of element");
    }
    vector<int> inplanedirect;
    {
      set <int> inplanedirectset;
      for(int i=0;i<3;++i)
      {
         inplanedirectset.insert(i);
      }
      inplanedirectset.erase(elenormdirect);

      for(set<int>::iterator id = inplanedirectset.begin();id!=inplanedirectset.end() ;++id)
      {
        inplanedirect.push_back(*id);
      }
    }

    // allocate vector for shapefunctions
    LINALG::Matrix<iel,1> funct;
    // allocate vector for shapederivatives
    LINALG::Matrix<3,iel> deriv;
    // space for the jacobian
    LINALG::Matrix<3,3>   xjm;

    // get the quad9 gaussrule for the in plane integration
    const IntegrationPoints2D  intpoints(intrule_quad_9point);

    // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
    // (now we allow even more)
    double layershift=0;

    // loop all levels in element
    for(set<int>::const_iterator id = planesinele.begin();id!=planesinele.end() ;++id)
    {
      // reset temporary values
      double area=0;

      double ubar=0;
      double vbar=0;
      double wbar=0;
      double pbar=0;

      double usqbar=0;
      double vsqbar=0;
      double wsqbar=0;
      double uvbar =0;
      double uwbar =0;
      double vwbar =0;
      double psqbar=0;

      // get the integration point in wall normal direction
      double e[3];

      e[elenormdirect]=-1.0+shift+layershift;
      if(upsidedown)
      {
        e[elenormdirect]*=-1;
      }

      // start loop over integration points in layer
      for (int iquad=0;iquad<intpoints.nquad;iquad++)
      {
        // get the other gauss point coordinates
        for(int i=0;i<2;++i)
        {
          e[inplanedirect[i]]=intpoints.qxg[iquad][i];
        }

        // compute the shape function values
        shape_function_3D(funct,e[0],e[1],e[2],distype);

        shape_function_3D_deriv1(deriv,e[0],e[1],e[2],distype);

        // get transposed Jacobian matrix and determinant
        //
        //        +-            -+ T      +-            -+
        //        | dx   dx   dx |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dr   dr   dr |
        //        |              |        |              |
        //        | dy   dy   dy |        | dx   dy   dz |
        //        | --   --   -- |   =    | --   --   -- |
        //        | dr   ds   dt |        | ds   ds   ds |
        //        |              |        |              |
        //        | dz   dz   dz |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dt   dt   dt |
        //        +-            -+        +-            -+
        //
        // The Jacobian is computed using the formula
        //
        //            +-----
        //   dx_j(r)   \      dN_k(r)
        //   -------  = +     ------- * (x_j)_k
        //    dr_i     /       dr_i       |
        //            +-----    |         |
        //            node k    |         |
        //                  derivative    |
        //                   of shape     |
        //                   function     |
        //                           component of
        //                          node coordinate
        //
        xjm.MultiplyNT(deriv,xyze);

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)
        if(xjm(normdirect,normdirect)<0)
        {
          xjm(normdirect,normdirect)=-1.0;
        }
        else
        {
          xjm(normdirect,normdirect)= 1.0;
        }

        const double det =
          xjm(0,0)*xjm(1,1)*xjm(2,2)
          +
          xjm(0,1)*xjm(1,2)*xjm(2,0)
          +
          xjm(0,2)*xjm(1,0)*xjm(2,1)
          -
          xjm(0,2)*xjm(1,1)*xjm(2,0)
          -
          xjm(0,0)*xjm(1,2)*xjm(2,1)
          -
          xjm(0,1)*xjm(1,0)*xjm(2,2);

        // check for degenerated elements
        if (det <= 0.0)
        {
          dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", Id(), det);
        }

#ifdef DEBUG
	// check whether this gausspoint is really inside the desired plane
	{
	  double x[3];
	  x[0]=0;
	  x[1]=0;
	  x[2]=0;
	  for(int inode=0;inode<iel;inode++)
	  {
	    x[0]+=funct(inode)*xyze(0,inode);
	    x[1]+=funct(inode)*xyze(1,inode);
	    x[2]+=funct(inode)*xyze(2,inode);
	  }

	  if(abs(x[normdirect]-(*planes)[*id])>2e-9)
	  {
	    dserror("Mixing up element cut planes during integration");
	  }
	}
#endif

        //interpolated values at gausspoints
        double ugp=0;
        double vgp=0;
        double wgp=0;
        double pgp=0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac=det*intpoints.qwgt[iquad];

        // increase area of cutting plane in element
        area += fac;

        for(int inode=0;inode<iel;inode++)
        {
          int finode=inode*4;

          ugp  += funct(inode)*sol(finode++);
          vgp  += funct(inode)*sol(finode++);
          wgp  += funct(inode)*sol(finode++);
          pgp  += funct(inode)*sol(finode  );
        }

        // add contribution to integral

        double dubar  = ugp*fac;
        double dvbar  = vgp*fac;
        double dwbar  = wgp*fac;
        double dpbar  = pgp*fac;

        ubar   += dubar;
        vbar   += dvbar;
        wbar   += dwbar;
        pbar   += dpbar;

        usqbar += ugp*dubar;
        vsqbar += vgp*dvbar;
        wsqbar += wgp*dwbar;
        uvbar  += ugp*dvbar;
        uwbar  += ugp*dwbar;
        vwbar  += vgp*dwbar;
        psqbar += pgp*dpbar;
      } // end loop integration points

      // add increments from this layer to processor local vectors
      (*sumarea)[*id] += area;

      (*sumu   )[*id] += ubar;
      (*sumv   )[*id] += vbar;
      (*sumw   )[*id] += wbar;
      (*sump   )[*id] += pbar;

      (*sumsqu )[*id] += usqbar;
      (*sumsqv )[*id] += vsqbar;
      (*sumsqw )[*id] += wsqbar;
      (*sumuv  )[*id] += uvbar;
      (*sumuw  )[*id] += uwbar;
      (*sumvw  )[*id] += vwbar;
      (*sumsqp )[*id] += psqbar;

      // jump to the next layer in the element.
      // in case of an hex8 element, the two coordinates are -1 and 1(+2)
      // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

      layershift+=2.0/(static_cast<double>(numplanesinele-1));
    }
  }
  else if(distype == DRT::Element::nurbs8 || distype == DRT::Element::nurbs27)
  {
    // get size of planecoords
    int size = planes->size();

    DRT::NURBS::NurbsDiscretization* nurbsdis
      =
      dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(discretization));

    if(nurbsdis == NULL)
    {
      dserror("we need a nurbs discretisation for nurbs elements\n");
    }

    // get nurbs dis' element numbers
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(0));

    // use size of planes and mele to determine number of layers
    int numsublayers=(size-1)/nele_x_mele_x_lele[1];

    // get the knotvector itself
    RCP<DRT::NURBS::Knotvector> knots=nurbsdis->GetKnotVector();

    DRT::Node**   nodes = Nodes();

    // get gid, location in the patch
    int gid = Id();

    vector<int> ele_cart_id(3);

    int npatch = -1;

    knots->ConvertEleGidToKnotIds(gid,npatch,ele_cart_id);
    if(npatch!=0)
    {
      dserror("expected single patch nurbs problem for calculating means");
    }

    // access elements knot span
    std::vector<Epetra_SerialDenseVector> eleknots(3);

    bool zero_size = false;
    zero_size = knots->GetEleKnots(eleknots,gid);

    // if we have a zero sized element due to a interpolated
    // point --- exit here
    if(zero_size)
    {
      return;
    }

    // aquire weights from nodes
    LINALG::Matrix<iel,1> weights;

    for (int inode=0; inode<iel; ++inode)
    {
      DRT::NURBS::ControlPoint* cp
	=
	dynamic_cast<DRT::NURBS::ControlPoint* > (nodes[inode]);

      weights(inode) = cp->W();
    }

    // get shapefunctions, compute all visualisation point positions
    LINALG::Matrix<iel,1> nurbs_shape_funct;
    // allocate vector for shapederivatives
    LINALG::Matrix<3,iel> nurbs_shape_deriv;
    // space for the jacobian
    LINALG::Matrix<3,3>   xjm;

    // there's one additional plane for the last element layer
    int endlayer=0;
    if(ele_cart_id[1]!=nele_x_mele_x_lele[1]-1)
    {
      endlayer=numsublayers;
    }
    else
    {
      endlayer=numsublayers+1;
    }

    // loop layers in element
    for(int rr=0;rr<endlayer;++rr)
    {
      // set gauss point coordinates
      Epetra_SerialDenseVector gp(3);

      gp(1)=-1.0+rr*2.0/((double)numsublayers);

      // get the quad9 gaussrule for the in plane integration
      const IntegrationPoints2D  intpoints(intrule_quad_9point);

      // reset temporary values
      double area=0;

      double ubar=0;
      double vbar=0;
      double wbar=0;
      double pbar=0;

      double usqbar=0;
      double vsqbar=0;
      double wsqbar=0;
      double uvbar =0;
      double uwbar =0;
      double vwbar =0;
      double psqbar=0;


      // start loop over integration points in layer
      for (int iquad=0;iquad<intpoints.nquad;iquad++)
      {

        // get the other gauss point coordinates
        gp(0)=intpoints.qxg[iquad][0];
        gp(2)=intpoints.qxg[iquad][1];

        // compute the shape function values
        DRT::NURBS::UTILS::nurbs_get_3D_funct_deriv
          (nurbs_shape_funct,
           nurbs_shape_deriv,
           gp               ,
           eleknots         ,
           weights          ,
           distype          );

        // get transposed Jacobian matrix and determinant
        //
        //        +-            -+ T      +-            -+
        //        | dx   dx   dx |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dr   dr   dr |
        //        |              |        |              |
        //        | dy   dy   dy |        | dx   dy   dz |
        //        | --   --   -- |   =    | --   --   -- |
        //        | dr   ds   dt |        | ds   ds   ds |
        //        |              |        |              |
        //        | dz   dz   dz |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dt   dt   dt |
        //        +-            -+        +-            -+
        //
        // The Jacobian is computed using the formula
        //
        //            +-----
        //   dx_j(r)   \      dN_k(r)
        //   -------  = +     ------- * (x_j)_k
        //    dr_i     /       dr_i       |
        //            +-----    |         |
        //            node k    |         |
        //                  derivative    |
        //                   of shape     |
        //                   function     |
        //                           component of
        //                          node coordinate
        //
        xjm.MultiplyNT(nurbs_shape_deriv,xyze);

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)
        if(xjm(normdirect,normdirect)<0)
        {
          xjm(normdirect,normdirect)=-1.0;
        }
        else
        {
          xjm(normdirect,normdirect)= 1.0;
        }

        const double det =
          xjm(0,0)*xjm(1,1)*xjm(2,2)
          +
          xjm(0,1)*xjm(1,2)*xjm(2,0)
          +
          xjm(0,2)*xjm(1,0)*xjm(2,1)
          -
          xjm(0,2)*xjm(1,1)*xjm(2,0)
          -
          xjm(0,0)*xjm(1,2)*xjm(2,1)
          -
          xjm(0,1)*xjm(1,0)*xjm(2,2);

        // check for degenerated elements
        if (det <= 0.0)
        {
          dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", Id(), det);
        }

        //interpolated values at gausspoints
        double ugp=0;
        double vgp=0;
        double wgp=0;
        double pgp=0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac=det*intpoints.qwgt[iquad];

        // increase area of cutting plane in element
        area += fac;

        for(int inode=0;inode<iel;inode++)
        {
          ugp += nurbs_shape_funct(inode)*sol(inode*4  );
          vgp += nurbs_shape_funct(inode)*sol(inode*4+1);
          wgp += nurbs_shape_funct(inode)*sol(inode*4+2);
          pgp += nurbs_shape_funct(inode)*sol(inode*4+3);
        }

        // add contribution to integral
        ubar   += ugp*fac;
        vbar   += vgp*fac;
        wbar   += wgp*fac;
        pbar   += pgp*fac;

        usqbar += ugp*ugp*fac;
        vsqbar += vgp*vgp*fac;
        wsqbar += wgp*wgp*fac;
        uvbar  += ugp*vgp*fac;
        uwbar  += ugp*wgp*fac;
        vwbar  += vgp*wgp*fac;
        psqbar += pgp*pgp*fac;
      } // end loop integration points


      // add increments from this layer to processor local vectors
      (*sumarea)[ele_cart_id[1]*numsublayers+rr] += area;

      (*sumu   )[ele_cart_id[1]*numsublayers+rr] += ubar;
      (*sumv   )[ele_cart_id[1]*numsublayers+rr] += vbar;
      (*sumw   )[ele_cart_id[1]*numsublayers+rr] += wbar;
      (*sump   )[ele_cart_id[1]*numsublayers+rr] += pbar;

      (*sumsqu )[ele_cart_id[1]*numsublayers+rr] += usqbar;
      (*sumsqv )[ele_cart_id[1]*numsublayers+rr] += vsqbar;
      (*sumsqw )[ele_cart_id[1]*numsublayers+rr] += wsqbar;
      (*sumuv  )[ele_cart_id[1]*numsublayers+rr] += uvbar;
      (*sumuw  )[ele_cart_id[1]*numsublayers+rr] += uwbar;
      (*sumvw  )[ele_cart_id[1]*numsublayers+rr] += vwbar;
      (*sumsqp )[ele_cart_id[1]*numsublayers+rr] += psqbar;
    }

  }
  else
  {
    dserror("Unknown element type for mean value evaluation\n");
  }
  return;
} // DRT::ELEMENTS::Fluid::f3_calc_means

/*---------------------------------------------------------------------*
 | Calculate spatial mean values for variable-density                  |
 | channel flow at low Mach number                           vg 02/09  |
 *---------------------------------------------------------------------*/
template<int iel>
void DRT::ELEMENTS::Fluid::f3_calc_loma_means(DRT::Discretization&  discretization,
                                               vector<double>&       velocitypressure,
                                               vector<double>&       temperature,
                                               Teuchos::ParameterList&        params,
                                               const double          eosfac
  )
{
  // get view of solution vector
  LINALG::Matrix<4*iel,1> velpre(&(velocitypressure[0]),true);
  LINALG::Matrix<4*iel,1> temp(&(temperature[0]),true);

  // set element data
  const DiscretizationType distype = this->Shape();

  // the plane normal tells you in which plane the integration takes place
  const int normdirect = params.get<int>("normal direction to homogeneous plane");

  // the vector planes contains the coordinates of the homogeneous planes (in
  // wall normal direction)
  RCP<vector<double> > planes = params.get<RCP<vector<double> > >("coordinate vector for hom. planes");

  // get the pointers to the solution vectors
  RCP<vector<double> > sumarea= params.get<RCP<vector<double> > >("element layer area");

  RCP<vector<double> > sumu   = params.get<RCP<vector<double> > >("mean velocity u");
  RCP<vector<double> > sumv   = params.get<RCP<vector<double> > >("mean velocity v");
  RCP<vector<double> > sumw   = params.get<RCP<vector<double> > >("mean velocity w");
  RCP<vector<double> > sump   = params.get<RCP<vector<double> > >("mean pressure p");
  RCP<vector<double> > sumrho = params.get<RCP<vector<double> > >("mean density rho");
  RCP<vector<double> > sumT   = params.get<RCP<vector<double> > >("mean temperature T");
  RCP<vector<double> > sumrhou  = params.get<RCP<vector<double> > >("mean momentum rho*u");
  RCP<vector<double> > sumrhouT = params.get<RCP<vector<double> > >("mean rho*u*T");

  RCP<vector<double> > sumsqu = params.get<RCP<vector<double> > >("mean value u^2");
  RCP<vector<double> > sumsqv = params.get<RCP<vector<double> > >("mean value v^2");
  RCP<vector<double> > sumsqw = params.get<RCP<vector<double> > >("mean value w^2");
  RCP<vector<double> > sumsqp = params.get<RCP<vector<double> > >("mean value p^2");
  RCP<vector<double> > sumsqrho = params.get<RCP<vector<double> > >("mean value rho^2");
  RCP<vector<double> > sumsqT = params.get<RCP<vector<double> > >("mean value T^2");

  RCP<vector<double> > sumuv  = params.get<RCP<vector<double> > >("mean value uv");
  RCP<vector<double> > sumuw  = params.get<RCP<vector<double> > >("mean value uw");
  RCP<vector<double> > sumvw  = params.get<RCP<vector<double> > >("mean value vw");
  RCP<vector<double> > sumuT  = params.get<RCP<vector<double> > >("mean value uT");
  RCP<vector<double> > sumvT  = params.get<RCP<vector<double> > >("mean value vT");
  RCP<vector<double> > sumwT  = params.get<RCP<vector<double> > >("mean value wT");

  // get node coordinates of element
  LINALG::Matrix<3,iel>  xyze;
  DRT::Node** nodes = Nodes();
  for(int inode=0;inode<iel;inode++)
  {
    const double* x = nodes[inode]->X();
    xyze(0,inode)=x[0];
    xyze(1,inode)=x[1];
    xyze(2,inode)=x[2];
  }

  if(distype == DRT::Element::hex8
     ||
     distype == DRT::Element::hex27
     ||
     distype == DRT::Element::hex20)
  {
    double min = xyze(normdirect,0);
    double max = xyze(normdirect,0);

    // set maximum and minimum value in wall normal direction
    for(int inode=0;inode<iel;inode++)
    {
      if(min > xyze(normdirect,inode)) min=xyze(normdirect,inode);
      if(max < xyze(normdirect,inode)) max=xyze(normdirect,inode);
    }

    // determine the ids of the homogeneous planes intersecting this element
    set<int> planesinele;
    for(unsigned nplane=0;nplane<planes->size();++nplane)
    {
    // get all available wall normal coordinates
      for(int nn=0;nn<iel;++nn)
      {
        if (min-2e-9 < (*planes)[nplane] && max+2e-9 > (*planes)[nplane])
          planesinele.insert(nplane);
      }
    }

    // remove lowest layer from planesinele to avoid double calculations. This is not done
    // for the first level (index 0) --- if deleted, shift the first integration point in
    // wall normal direction
    // the shift depends on the number of sampling planes in the element
    double shift=0;

    // set the number of planes which cut the element
    const int numplanesinele = planesinele.size();

    if(*planesinele.begin() != 0)
    {
      // this is not an element of the lowest element layer
      planesinele.erase(planesinele.begin());

      shift=2.0/(static_cast<double>(numplanesinele-1));
    }
    else
    {
      // this is an element of the lowest element layer. Increase the counter
      // in order to compute the total number of elements in one layer
      int* count = params.get<int*>("count processed elements");

      (*count)++;
    }

    // determine the orientation of the rst system compared to the xyz system
    int elenormdirect=-1;
    bool upsidedown =false;
    // the only thing of interest is how normdirect is oriented in the
    // element coordinate system
    if(xyze(normdirect,4)-xyze(normdirect,0)>2e-9)
    {
      // t aligned
      elenormdirect =2;
      cout << "upsidedown false" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)>2e-9)
    {
      // s aligned
      elenormdirect =1;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)>2e-9)
    {
      // r aligned
      elenormdirect =0;
    }
    else if(xyze(normdirect,4)-xyze(normdirect,0)<-2e-9)
    {
      cout << xyze(normdirect,4)-xyze(normdirect,0) << &endl;
      // -t aligned
      elenormdirect =2;
      upsidedown =true;
      cout << "upsidedown true" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)<-2e-9)
    {
      // -s aligned
      elenormdirect =1;
      upsidedown =true;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)<-2e-9)
    {
      // -r aligned
      elenormdirect =0;
      upsidedown =true;
    }
    else
    {
      dserror("cannot determine orientation of plane normal in local coordinate system of element");
    }
    vector<int> inplanedirect;
    {
      set <int> inplanedirectset;
      for(int i=0;i<3;++i)
      {
        inplanedirectset.insert(i);
      }
      inplanedirectset.erase(elenormdirect);

      for(set<int>::iterator id = inplanedirectset.begin();id!=inplanedirectset.end() ;++id)
      {
        inplanedirect.push_back(*id);
      }
    }

    // allocate vector for shapefunctions
    LINALG::Matrix<iel,1> funct;
    // allocate vector for shapederivatives
    LINALG::Matrix<3,iel> deriv;
    // space for the jacobian
    LINALG::Matrix<3,3>   xjm;

    // get the quad9 gaussrule for the in-plane integration
    const IntegrationPoints2D  intpoints(intrule_quad_9point);

    // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
    // (now we allow even more)
    double layershift=0;

    // loop all levels in element
    for(set<int>::const_iterator id = planesinele.begin();id!=planesinele.end() ;++id)
    {
      // reset temporary values
      double area=0;

      double ubar=0;
      double vbar=0;
      double wbar=0;
      double pbar=0;
      double rhobar=0;
      double Tbar=0;
      double rhoubar=0;
      double rhouTbar=0;

      double usqbar=0;
      double vsqbar=0;
      double wsqbar=0;
      double psqbar=0;
      double rhosqbar=0;
      double Tsqbar=0;

      double uvbar =0;
      double uwbar =0;
      double vwbar =0;
      double uTbar =0;
      double vTbar =0;
      double wTbar =0;

      // get the integration point in wall normal direction
      double e[3];

      e[elenormdirect]=-1.0+shift+layershift;
      if(upsidedown) e[elenormdirect]*=-1;

      // start loop over integration points in layer
      for (int iquad=0;iquad<intpoints.nquad;iquad++)
      {
        // get the other gauss point coordinates
        for(int i=0;i<2;++i)
        {
          e[inplanedirect[i]]=intpoints.qxg[iquad][i];
        }

        // compute the shape function values
        shape_function_3D(funct,e[0],e[1],e[2],distype);
        shape_function_3D_deriv1(deriv,e[0],e[1],e[2],distype);

        // get transposed Jacobian matrix and determinant
        //
        //        +-            -+ T      +-            -+
        //        | dx   dx   dx |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dr   dr   dr |
        //        |              |        |              |
        //        | dy   dy   dy |        | dx   dy   dz |
        //        | --   --   -- |   =    | --   --   -- |
        //        | dr   ds   dt |        | ds   ds   ds |
        //        |              |        |              |
        //        | dz   dz   dz |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dt   dt   dt |
        //        +-            -+        +-            -+
        //
        // The Jacobian is computed using the formula
        //
        //            +-----
        //   dx_j(r)   \      dN_k(r)
        //   -------  = +     ------- * (x_j)_k
        //    dr_i     /       dr_i       |
        //            +-----    |         |
        //            node k    |         |
        //                  derivative    |
        //                   of shape     |
        //                   function     |
        //                           component of
        //                          node coordinate
        //
        xjm.MultiplyNT(deriv,xyze);

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)
        if (xjm(normdirect,normdirect)<0) xjm(normdirect,normdirect)=-1.0;
        else                              xjm(normdirect,normdirect)= 1.0;

        const double det =
          xjm(0,0)*xjm(1,1)*xjm(2,2)
          +
          xjm(0,1)*xjm(1,2)*xjm(2,0)
          +
          xjm(0,2)*xjm(1,0)*xjm(2,1)
          -
          xjm(0,2)*xjm(1,1)*xjm(2,0)
          -
          xjm(0,0)*xjm(1,2)*xjm(2,1)
          -
          xjm(0,1)*xjm(1,0)*xjm(2,2);

        // check for degenerated elements
        if (det <= 0.0) dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", Id(), det);

        //interpolated values at gausspoints
        double ugp=0;
        double vgp=0;
        double wgp=0;
        double pgp=0;
        double rhogp=0;
        double Tgp=0;
        double rhougp=0;
        double rhouTgp=0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac=det*intpoints.qwgt[iquad];

        // increase area of cutting plane in element
        area += fac;

        for(int inode=0;inode<iel;inode++)
        {
          int finode=inode*4;

          ugp   += funct(inode)*velpre(finode++);
          vgp   += funct(inode)*velpre(finode++);
          wgp   += funct(inode)*velpre(finode++);
          pgp   += funct(inode)*velpre(finode  );
          Tgp   += funct(inode)*temp(finode  );
        }
        rhogp   = eosfac/Tgp;
        rhouTgp = eosfac*ugp;
        rhougp  = rhouTgp/Tgp;

        // add contribution to integral
        double dubar   = ugp*fac;
        double dvbar   = vgp*fac;
        double dwbar   = wgp*fac;
        double dpbar   = pgp*fac;
        double drhobar = rhogp*fac;
        double dTbar   = Tgp*fac;
        double drhoubar  = rhougp*fac;
        double drhouTbar = rhouTgp*fac;

        ubar   += dubar;
        vbar   += dvbar;
        wbar   += dwbar;
        pbar   += dpbar;
        rhobar += drhobar;
        Tbar   += dTbar;
        rhoubar  += drhoubar;
        rhouTbar += drhouTbar;

        usqbar   += ugp*dubar;
        vsqbar   += vgp*dvbar;
        wsqbar   += wgp*dwbar;
        psqbar   += pgp*dpbar;
        rhosqbar += rhogp*drhobar;
        Tsqbar   += Tgp*dTbar;

        uvbar  += ugp*dvbar;
        uwbar  += ugp*dwbar;
        vwbar  += vgp*dwbar;
        uTbar  += ugp*dTbar;
        vTbar  += vgp*dTbar;
        wTbar  += wgp*dTbar;
      } // end loop integration points

      // add increments from this layer to processor local vectors
      (*sumarea)[*id] += area;

      (*sumu   )[*id] += ubar;
      (*sumv   )[*id] += vbar;
      (*sumw   )[*id] += wbar;
      (*sump   )[*id] += pbar;
      (*sumrho )[*id] += rhobar;
      (*sumT   )[*id] += Tbar;
      (*sumrhou)[*id] += rhoubar;
      (*sumrhouT)[*id] += rhouTbar;

      (*sumsqu  )[*id] += usqbar;
      (*sumsqv  )[*id] += vsqbar;
      (*sumsqw  )[*id] += wsqbar;
      (*sumsqp  )[*id] += psqbar;
      (*sumsqrho)[*id] += rhosqbar;
      (*sumsqT  )[*id] += Tsqbar;

      (*sumuv  )[*id] += uvbar;
      (*sumuw  )[*id] += uwbar;
      (*sumvw  )[*id] += vwbar;
      (*sumuT  )[*id] += uTbar;
      (*sumvT  )[*id] += vTbar;
      (*sumwT  )[*id] += wTbar;

      // jump to the next layer in the element.
      // in case of an hex8 element, the two coordinates are -1 and 1(+2)
      // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

      layershift+=2.0/(static_cast<double>(numplanesinele-1));
    }
  }
  else
    dserror("Unknown element type for low-Mach-number mean value evaluation\n");

  return;
} // DRT::ELEMENTS::Fluid::f3_calc_loma_means


/*---------------------------------------------------------------------*
 | Calculate spatial mean values for passive scalar                    |
 | transport in turbulent channel flow                 rasthofer 01/12 |
 *---------------------------------------------------------------------*/
template<int iel>
void DRT::ELEMENTS::Fluid::f3_calc_scatra_means(DRT::Discretization&  discretization,
                                                 vector<double>&       velocitypressure,
                                                 vector<double>&       scalar,
                                                 Teuchos::ParameterList&        params)
{
  // get view of solution vector
  LINALG::Matrix<4*iel,1> velpre(&(velocitypressure[0]),true);
  LINALG::Matrix<4*iel,1> phi(&(scalar[0]),true);

  // set element data
  const DiscretizationType distype = this->Shape();

  // the plane normal tells you in which plane the integration takes place
  const int normdirect = params.get<int>("normal direction to homogeneous plane");

  // the vector planes contains the coordinates of the homogeneous planes (in
  // wall normal direction)
  RCP<vector<double> > planes = params.get<RCP<vector<double> > >("coordinate vector for hom. planes");

  // get the pointers to the solution vectors
  RCP<vector<double> > sumarea= params.get<RCP<vector<double> > >("element layer area");

  RCP<vector<double> > sumu   = params.get<RCP<vector<double> > >("mean velocity u");
  RCP<vector<double> > sumv   = params.get<RCP<vector<double> > >("mean velocity v");
  RCP<vector<double> > sumw   = params.get<RCP<vector<double> > >("mean velocity w");
  RCP<vector<double> > sump   = params.get<RCP<vector<double> > >("mean pressure p");
  RCP<vector<double> > sumphi   = params.get<RCP<vector<double> > >("mean scalar phi");

  RCP<vector<double> > sumsqu = params.get<RCP<vector<double> > >("mean value u^2");
  RCP<vector<double> > sumsqv = params.get<RCP<vector<double> > >("mean value v^2");
  RCP<vector<double> > sumsqw = params.get<RCP<vector<double> > >("mean value w^2");
  RCP<vector<double> > sumsqp = params.get<RCP<vector<double> > >("mean value p^2");
  RCP<vector<double> > sumsqphi = params.get<RCP<vector<double> > >("mean value phi^2");

  RCP<vector<double> > sumuv  = params.get<RCP<vector<double> > >("mean value uv");
  RCP<vector<double> > sumuw  = params.get<RCP<vector<double> > >("mean value uw");
  RCP<vector<double> > sumvw  = params.get<RCP<vector<double> > >("mean value vw");
  RCP<vector<double> > sumuphi  = params.get<RCP<vector<double> > >("mean value uphi");
  RCP<vector<double> > sumvphi  = params.get<RCP<vector<double> > >("mean value vphi");
  RCP<vector<double> > sumwphi  = params.get<RCP<vector<double> > >("mean value wphi");

  // get node coordinates of element
  LINALG::Matrix<3,iel>  xyze;
  DRT::Node** nodes = Nodes();
  for(int inode=0;inode<iel;inode++)
  {
    const double* x = nodes[inode]->X();
    xyze(0,inode)=x[0];
    xyze(1,inode)=x[1];
    xyze(2,inode)=x[2];
  }

  if(distype == DRT::Element::hex8
     ||
     distype == DRT::Element::hex27
     ||
     distype == DRT::Element::hex20)
  {
    double min = xyze(normdirect,0);
    double max = xyze(normdirect,0);

    // set maximum and minimum value in wall normal direction
    for(int inode=0;inode<iel;inode++)
    {
      if(min > xyze(normdirect,inode)) min=xyze(normdirect,inode);
      if(max < xyze(normdirect,inode)) max=xyze(normdirect,inode);
    }

    // determine the ids of the homogeneous planes intersecting this element
    set<int> planesinele;
    for(unsigned nplane=0;nplane<planes->size();++nplane)
    {
    // get all available wall normal coordinates
      for(int nn=0;nn<iel;++nn)
      {
        if (min-2e-9 < (*planes)[nplane] && max+2e-9 > (*planes)[nplane])
          planesinele.insert(nplane);
      }
    }

    // remove lowest layer from planesinele to avoid double calculations. This is not done
    // for the first level (index 0) --- if deleted, shift the first integration point in
    // wall normal direction
    // the shift depends on the number of sampling planes in the element
    double shift=0;

    // set the number of planes which cut the element
    const int numplanesinele = planesinele.size();

    if(*planesinele.begin() != 0)
    {
      // this is not an element of the lowest element layer
      planesinele.erase(planesinele.begin());

      shift=2.0/(static_cast<double>(numplanesinele-1));
    }
    else
    {
      // this is an element of the lowest element layer. Increase the counter
      // in order to compute the total number of elements in one layer
      int* count = params.get<int*>("count processed elements");

      (*count)++;
    }

    // determine the orientation of the rst system compared to the xyz system
    int elenormdirect=-1;
    bool upsidedown =false;
    // the only thing of interest is how normdirect is oriented in the
    // element coordinate system
    if(xyze(normdirect,4)-xyze(normdirect,0)>2e-9)
    {
      // t aligned
      elenormdirect =2;
      cout << "upsidedown false" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)>2e-9)
    {
      // s aligned
      elenormdirect =1;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)>2e-9)
    {
      // r aligned
      elenormdirect =0;
    }
    else if(xyze(normdirect,4)-xyze(normdirect,0)<-2e-9)
    {
      cout << xyze(normdirect,4)-xyze(normdirect,0) << &endl;
      // -t aligned
      elenormdirect =2;
      upsidedown =true;
      cout << "upsidedown true" <<&endl;
    }
    else if (xyze(normdirect,3)-xyze(normdirect,0)<-2e-9)
    {
      // -s aligned
      elenormdirect =1;
      upsidedown =true;
    }
    else if (xyze(normdirect,1)-xyze(normdirect,0)<-2e-9)
    {
      // -r aligned
      elenormdirect =0;
      upsidedown =true;
    }
    else
    {
      dserror("cannot determine orientation of plane normal in local coordinate system of element");
    }
    vector<int> inplanedirect;
    {
      set <int> inplanedirectset;
      for(int i=0;i<3;++i)
      {
        inplanedirectset.insert(i);
      }
      inplanedirectset.erase(elenormdirect);

      for(set<int>::iterator id = inplanedirectset.begin();id!=inplanedirectset.end() ;++id)
      {
        inplanedirect.push_back(*id);
      }
    }

    // allocate vector for shapefunctions
    LINALG::Matrix<iel,1> funct;
    // allocate vector for shapederivatives
    LINALG::Matrix<3,iel> deriv;
    // space for the jacobian
    LINALG::Matrix<3,3>   xjm;

    // get the quad9 gaussrule for the in-plane integration
    const IntegrationPoints2D  intpoints(intrule_quad_9point);

    // a hex8 element has two levels, the hex20 and hex27 element have three layers to sample
    // (now we allow even more)
    double layershift=0;

    // loop all levels in element
    for(set<int>::const_iterator id = planesinele.begin();id!=planesinele.end() ;++id)
    {
      // reset temporary values
      double area=0;

      double ubar=0;
      double vbar=0;
      double wbar=0;
      double pbar=0;
      double phibar=0;

      double usqbar=0;
      double vsqbar=0;
      double wsqbar=0;
      double psqbar=0;
      double phisqbar=0;

      double uvbar =0;
      double uwbar =0;
      double vwbar =0;
      double uphibar =0;
      double vphibar =0;
      double wphibar =0;

      // get the integration point in wall normal direction
      double e[3];

      e[elenormdirect]=-1.0+shift+layershift;
      if(upsidedown) e[elenormdirect]*=-1;

      // start loop over integration points in layer
      for (int iquad=0;iquad<intpoints.nquad;iquad++)
      {
        // get the other gauss point coordinates
        for(int i=0;i<2;++i)
        {
          e[inplanedirect[i]]=intpoints.qxg[iquad][i];
        }

        // compute the shape function values
        shape_function_3D(funct,e[0],e[1],e[2],distype);
        shape_function_3D_deriv1(deriv,e[0],e[1],e[2],distype);

        // get transposed Jacobian matrix and determinant
        //
        //        +-            -+ T      +-            -+
        //        | dx   dx   dx |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dr   dr   dr |
        //        |              |        |              |
        //        | dy   dy   dy |        | dx   dy   dz |
        //        | --   --   -- |   =    | --   --   -- |
        //        | dr   ds   dt |        | ds   ds   ds |
        //        |              |        |              |
        //        | dz   dz   dz |        | dx   dy   dz |
        //        | --   --   -- |        | --   --   -- |
        //        | dr   ds   dt |        | dt   dt   dt |
        //        +-            -+        +-            -+
        //
        // The Jacobian is computed using the formula
        //
        //            +-----
        //   dx_j(r)   \      dN_k(r)
        //   -------  = +     ------- * (x_j)_k
        //    dr_i     /       dr_i       |
        //            +-----    |         |
        //            node k    |         |
        //                  derivative    |
        //                   of shape     |
        //                   function     |
        //                           component of
        //                          node coordinate
        //
        xjm.MultiplyNT(deriv,xyze);

        // we assume that every plane parallel to the wall is preserved
        // hence we can compute the jacobian determinant of the 2d cutting
        // element by replacing max-min by one on the diagonal of the
        // jacobi matrix (the two non-diagonal elements are zero)
        if (xjm(normdirect,normdirect)<0) xjm(normdirect,normdirect)=-1.0;
        else                              xjm(normdirect,normdirect)= 1.0;

        const double det =
          xjm(0,0)*xjm(1,1)*xjm(2,2)
          +
          xjm(0,1)*xjm(1,2)*xjm(2,0)
          +
          xjm(0,2)*xjm(1,0)*xjm(2,1)
          -
          xjm(0,2)*xjm(1,1)*xjm(2,0)
          -
          xjm(0,0)*xjm(1,2)*xjm(2,1)
          -
          xjm(0,1)*xjm(1,0)*xjm(2,2);

        // check for degenerated elements
        if (det <= 0.0) dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", Id(), det);

        //interpolated values at gausspoints
        double ugp=0;
        double vgp=0;
        double wgp=0;
        double pgp=0;
        double phigp=0;

        // the computation of this jacobian determinant from the 3d
        // mapping is based on the assumption that we do not deform
        // our elements in wall normal direction!
        const double fac=det*intpoints.qwgt[iquad];

        // increase area of cutting plane in element
        area += fac;

        for(int inode=0;inode<iel;inode++)
        {
          int finode=inode*4;

          ugp   += funct(inode)*velpre(finode++);
          vgp   += funct(inode)*velpre(finode++);
          wgp   += funct(inode)*velpre(finode++);
          pgp   += funct(inode)*velpre(finode  );
          phigp   += funct(inode)*phi(finode  );
        }

        // add contribution to integral
        double dubar   = ugp*fac;
        double dvbar   = vgp*fac;
        double dwbar   = wgp*fac;
        double dpbar   = pgp*fac;
        double dphibar   = phigp*fac;

        ubar   += dubar;
        vbar   += dvbar;
        wbar   += dwbar;
        pbar   += dpbar;
        phibar += dphibar;

        usqbar   += ugp*dubar;
        vsqbar   += vgp*dvbar;
        wsqbar   += wgp*dwbar;
        psqbar   += pgp*dpbar;
        phisqbar   += phigp*dphibar;

        uvbar  += ugp*dvbar;
        uwbar  += ugp*dwbar;
        vwbar  += vgp*dwbar;
        uphibar  += ugp*dphibar;
        vphibar  += vgp*dphibar;
        wphibar  += wgp*dphibar;
      } // end loop integration points

      // add increments from this layer to processor local vectors
      (*sumarea)[*id] += area;

      (*sumu   )[*id] += ubar;
      (*sumv   )[*id] += vbar;
      (*sumw   )[*id] += wbar;
      (*sump   )[*id] += pbar;
      (*sumphi   )[*id] += phibar;

      (*sumsqu  )[*id] += usqbar;
      (*sumsqv  )[*id] += vsqbar;
      (*sumsqw  )[*id] += wsqbar;
      (*sumsqp  )[*id] += psqbar;
      (*sumsqphi  )[*id] += phisqbar;

      (*sumuv  )[*id] += uvbar;
      (*sumuw  )[*id] += uwbar;
      (*sumvw  )[*id] += vwbar;
      (*sumuphi  )[*id] += uphibar;
      (*sumvphi  )[*id] += vphibar;
      (*sumwphi  )[*id] += wphibar;

      // jump to the next layer in the element.
      // in case of an hex8 element, the two coordinates are -1 and 1(+2)
      // for quadratic elements with three sample planes, we have -1,0(+1),1(+2)

      layershift+=2.0/(static_cast<double>(numplanesinele-1));
    }
  }
  else
    dserror("Unknown element type for turbulent passive scalar mean value evaluation\n");

  return;
} // DRT::ELEMENTS::Fluid::f3_calc_scatra_means


//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
template<int iel>
void DRT::ELEMENTS::Fluid::f3_apply_box_filter(
    bool             dyn_smagorinsky,
    vector<double>&  myvel,
    double*          bvel_hat,
    double*          breystr_hat,
    double*          bmodeled_stress_grid_scale_hat,
    double&          volume
    )
{
  // alloc a fixed size array for nodal velocities
  LINALG::Matrix<3,iel>   evel;

  // wrap matrix objects in fixed-size arrays
  LINALG::Matrix<4*iel,1> myvelvec                     (&(myvel[0])                   ,true);
  LINALG::Matrix<3,1>     vel_hat                      (bvel_hat                      ,true);
  LINALG::Matrix<3,3>     reystr_hat                   (breystr_hat                   ,true);
  LINALG::Matrix<3,3>     modeled_stress_grid_scale_hat(bmodeled_stress_grid_scale_hat,true);

  // split velocity and throw away  pressure, insert into element array
  for (int i=0;i<iel;++i)
  {
    int fi =4*i;

    evel(0,i) = myvelvec(fi++);
    evel(1,i) = myvelvec(fi++);
    evel(2,i) = myvelvec(fi  );
  }

   // number of spatial dimensions is always 3
  const int NSD = 3;

  // set element data
  const DiscretizationType distype = this->Shape();

  // allocate arrays for shapefunctions, derivatives and the transposed jacobian
  LINALG::Matrix<iel,  1>  funct;
  LINALG::Matrix<NSD,NSD>  xjm  ;
  LINALG::Matrix<NSD,NSD>  xji  ;
  LINALG::Matrix<NSD,iel>  deriv;
  LINALG::Matrix<NSD,iel>  derxy;

  LINALG::Matrix<NSD,1> velint;
  LINALG::Matrix<NSD,NSD> vderxy;
  LINALG::Matrix<NSD,NSD> epsilon;
  double rateofstrain = 0.0;

  // get node coordinates of element
  LINALG::Matrix<NSD,iel> xyze;
  for(int inode=0;inode<iel;inode++)
  {
    xyze(0,inode)=Nodes()[inode]->X()[0];
    xyze(1,inode)=Nodes()[inode]->X()[1];
    xyze(2,inode)=Nodes()[inode]->X()[2];
  }

  // use one point gauss rule to calculate tau at element center
  DRT::UTILS::GaussRule3D integrationrule_filter=DRT::UTILS::intrule_hex_1point;
  switch (distype)
  {
      case DRT::Element::hex8:
        integrationrule_filter = DRT::UTILS::intrule_hex_1point;
        break;
      case DRT::Element::tet4:
        integrationrule_filter = DRT::UTILS::intrule_tet_1point;
        break;
      case DRT::Element::tet10:
      case DRT::Element::hex20:
      case DRT::Element::hex27:
        dserror("the box filtering operation is only permitted for linear elements\n");
        break;
      default:
        dserror("invalid discretization type for fluid3");
  }

  // gaussian points
  const DRT::UTILS::IntegrationPoints3D intpoints_onepoint(integrationrule_filter);

  // shape functions and derivs at element center
  const double e1    = intpoints_onepoint.qxg[0][0];
  const double e2    = intpoints_onepoint.qxg[0][1];
  const double e3    = intpoints_onepoint.qxg[0][2];
  const double wquad = intpoints_onepoint.qwgt[0];

  DRT::UTILS::shape_function_3D       (funct,e1,e2,e3,distype);
  DRT::UTILS::shape_function_3D_deriv1(deriv,e1,e2,e3,distype);

    // get Jacobian matrix and determinant
    for (int nn=0;nn<NSD;++nn)
    {
      for (int rr=0;rr<NSD;++rr)
      {
        xjm(nn,rr)=deriv(nn,0)*xyze(rr,0);
        for (int mm=1;mm<iel;++mm)
        {
          xjm(nn,rr)+=deriv(nn,mm)*xyze(rr,mm);
        }
      }
    }

    const double det = xjm(0,0)*xjm(1,1)*xjm(2,2)+
                       xjm(0,1)*xjm(1,2)*xjm(2,0)+
                       xjm(0,2)*xjm(1,0)*xjm(2,1)-
                       xjm(0,2)*xjm(1,1)*xjm(2,0)-
                       xjm(0,0)*xjm(1,2)*xjm(2,1)-
                       xjm(0,1)*xjm(1,0)*xjm(2,2);

    //
    //             compute global first derivates
    //
    //LINALG::Matrix<NSD,iel> derxy;

  if (dyn_smagorinsky)
  {
    /*
      Use the Jacobian and the known derivatives in element coordinate
      directions on the right hand side to compute the derivatives in
      global coordinate directions

            +-                 -+     +-    -+      +-    -+
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dr    dr    dr   |     |  dx  |      |  dr  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |  *  | ---- |   =  | ---- | for all k
            |  ds    ds    ds   |     |  dy  |      |  ds  |
            |                   |     |      |      |      |
            |  dx    dy    dz   |     | dN_k |      | dN_k |
            |  --    --    --   |     | ---- |      | ---- |
            |  dt    dt    dt   |     |  dz  |      |  dt  |
            +-                 -+     +-    -+      +-    -+

    */
    xji(0,0) = (  xjm(1,1)*xjm(2,2) - xjm(2,1)*xjm(1,2))/det;
    xji(1,0) = (- xjm(1,0)*xjm(2,2) + xjm(2,0)*xjm(1,2))/det;
    xji(2,0) = (  xjm(1,0)*xjm(2,1) - xjm(2,0)*xjm(1,1))/det;
    xji(0,1) = (- xjm(0,1)*xjm(2,2) + xjm(2,1)*xjm(0,2))/det;
    xji(1,1) = (  xjm(0,0)*xjm(2,2) - xjm(2,0)*xjm(0,2))/det;
    xji(2,1) = (- xjm(0,0)*xjm(2,1) + xjm(2,0)*xjm(0,1))/det;
    xji(0,2) = (  xjm(0,1)*xjm(1,2) - xjm(1,1)*xjm(0,2))/det;
    xji(1,2) = (- xjm(0,0)*xjm(1,2) + xjm(1,0)*xjm(0,2))/det;
    xji(2,2) = (  xjm(0,0)*xjm(1,1) - xjm(1,0)*xjm(0,1))/det;

    // compute global derivates
    for (int nn=0;nn<NSD;++nn)
    {
      for (int rr=0;rr<iel;++rr)
      {
        derxy(nn,rr)=deriv(0,rr)*xji(nn,0);
        for (int mm=1;mm<NSD;++mm)
        {
          derxy(nn,rr)+=deriv(mm,rr)*xji(nn,mm);
        }
      }
    }
  }

  // get velocities (n+alpha_F/1,i) at integration point
  //
  //                   +-----
  //       n+af/1       \                  n+af/1
  //    vel      (x) =   +      N (x) * vel
  //                    /        j         j
  //                   +-----
  //                   node j
  //
  //LINALG::Matrix<NSD,1> velint;
  for (int rr=0;rr<NSD;++rr)
  {
    velint(rr)=funct(0)*evel(rr,0);
    for (int mm=1;mm<iel;++mm)
    {
      velint(rr)+=funct(mm)*evel(rr,mm);
    }
  }

  if (dyn_smagorinsky)
  {
    // get velocity (n+alpha_F/1,i) derivatives at integration point
    //
    //       n+af/1      +-----  dN (x)
    //   dvel      (x)    \        k         n+af/1
    //   ------------- =   +     ------ * vel
    //        dx          /        dx        k
    //          j        +-----      j
    //                   node k
    //
    // j : direction of derivative x/y/z
    //
    // LINALG::Matrix<NSD,NSD> vderxy;
    for (int nn=0;nn<NSD;++nn)
    {
      for (int rr=0;rr<NSD;++rr)
      {
        vderxy(nn,rr)=derxy(rr,0)*evel(nn,0);
        for (int mm=1;mm<iel;++mm)
        {
          vderxy(nn,rr)+=derxy(rr,mm)*evel(nn,mm);
        }
      }
    }

    /*
                              +-     n+af/1          n+af/1    -+
            / h \       1.0   |  dvel_i    (x)   dvel_j    (x)  |
       eps | u   |    = --- * |  ------------- + -------------  |
            \   / ij    2.0   |       dx              dx        |
                              +-        j               i      -+
    */
    //LINALG::Matrix<NSD,NSD> epsilon;

    for (int nn=0;nn<NSD;++nn)
    {
      for (int rr=0;rr<NSD;++rr)
      {
        epsilon(nn,rr)=0.5*(vderxy(nn,rr)+vderxy(rr,nn));
      }
    }

    //
    // modeled part of subgrid scale stresses
    //
    /*    +-                                 -+ 1
          |          / h \           / h \    | -         / h \
          | 2 * eps | u   |   * eps | u   |   | 2  * eps | u   |
          |          \   / kl        \   / kl |           \   / ij
          +-                                 -+

          |                                   |
          +-----------------------------------+
               'resolved' rate of strain
    */

    //double rateofstrain = 0;

    for(int rr=0;rr<NSD;rr++)
    {
      for(int mm=0;mm<NSD;mm++)
      {
        rateofstrain += epsilon(rr,mm)*epsilon(rr,mm);
      }
    }
    rateofstrain *= 2.0;
    rateofstrain = sqrt(rateofstrain);
  }


  //--------------------------------------------------
  // one point integrations

  // determine contribution to patch volume
  volume = wquad*det;

  for (int rr=0;rr<NSD;++rr)
  {
    double temp=velint(rr)*volume;

    // add contribution to integral over velocities
    vel_hat(rr) += temp;

    // add contribution to integral over reynolds stresses
    for (int nn=0;nn<NSD;++nn)
    {
      reystr_hat(rr,nn) += temp*velint(nn);
    }
  }

  if (dyn_smagorinsky)
  {
    // add contribution to integral over the modeled part of subgrid
    // scale stresses
    double rateofstrain_volume=rateofstrain*volume;
    for (int rr=0;rr<NSD;++rr)
    {
      for (int nn=0;nn<NSD;++nn)
      {
        modeled_stress_grid_scale_hat(rr,nn) += rateofstrain_volume * epsilon(rr,nn);
      }
    }
  }

  return;
} // DRT::ELEMENTS::Fluid::f3_apply_box_filter


//----------------------------------------------------------------------
// Calculate the quantities LijMij and MijMij, to compare the influence
// of the modeled and resolved stress tensor --- from this relation, Cs
// will be computed
//----------------------------------------------------------------------
template<int iel>
void DRT::ELEMENTS::Fluid::f3_calc_smag_const_LijMij_and_MijMij(
  RCP<Epetra_MultiVector>& filtered_vel                       ,
  RCP<Epetra_MultiVector>& col_filtered_reynoldsstress        ,
  RCP<Epetra_MultiVector>& col_filtered_modeled_subgrid_stress,
  double&                  LijMij,
  double&                  MijMij,
  double&                  xcenter,
  double&                  ycenter,
  double&                  zcenter)
{

  LINALG::Matrix<3,iel> evel_hat                            ;
  LINALG::Matrix<9,iel> ereynoldsstress_hat                 ;
  LINALG::Matrix<9,iel> efiltered_modeled_subgrid_stress_hat;

  for (int nn=0;nn<iel;++nn)
  {
    int lid = (Nodes()[nn])->LID();

    for (int dimi=0;dimi<3;++dimi)
    {
      evel_hat(dimi,nn) = (*((*filtered_vel)(dimi)))[lid];

      for (int dimj=0;dimj<3;++dimj)
      {
        int index=3*dimi+dimj;

        ereynoldsstress_hat(index,nn)
          =(*((*col_filtered_reynoldsstress)(index)))[lid];

        efiltered_modeled_subgrid_stress_hat (index,nn)
          =(*((*col_filtered_modeled_subgrid_stress)(index)))[lid];
      }
    }
  }

  // set element data
  const DiscretizationType distype = this->Shape();

  // allocate arrays for shapefunctions, derivatives and the transposed jacobian
  LINALG::Matrix<iel,1> funct;
  LINALG::Matrix<3,iel> deriv;

  //this will be the of a point in the element interior
  xcenter = 0;
  ycenter = 0;
  zcenter = 0;

  // get node coordinates of element
  LINALG::Matrix<3,iel> xyze;
  for(int inode=0;inode<iel;inode++)
  {
    xyze(0,inode)=Nodes()[inode]->X()[0];
    xyze(1,inode)=Nodes()[inode]->X()[1];
    xyze(2,inode)=Nodes()[inode]->X()[2];

    xcenter+=xyze(1,inode);
    ycenter+=xyze(1,inode);
    zcenter+=xyze(1,inode);
  }
  xcenter/=iel;
  ycenter/=iel;
  zcenter/=iel;


  // use one point gauss rule to calculate tau at element center
  DRT::UTILS::GaussRule3D integrationrule_filter=DRT::UTILS::intrule_hex_1point;
  switch (distype)
  {
      case DRT::Element::hex8:
      case DRT::Element::hex20:
      case DRT::Element::hex27:
        integrationrule_filter = DRT::UTILS::intrule_hex_1point;
        break;
      case DRT::Element::tet4:
      case DRT::Element::tet10:
        integrationrule_filter = DRT::UTILS::intrule_tet_1point;
        break;
      default:
        dserror("invalid discretization type for fluid3");
  }

  // gaussian points --- i.e. the midpoint
  const DRT::UTILS::IntegrationPoints3D intpoints_onepoint(integrationrule_filter);
  const double e1    = intpoints_onepoint.qxg[0][0];
  const double e2    = intpoints_onepoint.qxg[0][1];
  const double e3    = intpoints_onepoint.qxg[0][2];

  // shape functions and derivs at element center
  DRT::UTILS::shape_function_3D       (funct,e1,e2,e3,distype);
  DRT::UTILS::shape_function_3D_deriv1(deriv,e1,e2,e3,distype);

  LINALG::Matrix<3,3> xjm;
  // get Jacobian matrix and its determinant
  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<3;++rr)
    {
      xjm(nn,rr)=deriv(nn,0)*xyze(rr,0);
      for (int mm=1;mm<iel;++mm)
      {
        xjm(nn,rr)+=deriv(nn,mm)*xyze(rr,mm);
      }
    }
  }
  const double det = xjm(0,0)*xjm(1,1)*xjm(2,2)+
                     xjm(0,1)*xjm(1,2)*xjm(2,0)+
                     xjm(0,2)*xjm(1,0)*xjm(2,1)-
                     xjm(0,2)*xjm(1,1)*xjm(2,0)-
                     xjm(0,0)*xjm(1,2)*xjm(2,1)-
                     xjm(0,1)*xjm(1,0)*xjm(2,2);

  //
  //             compute global first derivates
  //
  LINALG::Matrix<3,iel> derxy;
  /*
    Use the Jacobian and the known derivatives in element coordinate
    directions on the right hand side to compute the derivatives in
    global coordinate directions

          +-                 -+     +-    -+      +-    -+
          |  dx    dy    dz   |     | dN_k |      | dN_k |
          |  --    --    --   |     | ---- |      | ---- |
          |  dr    dr    dr   |     |  dx  |      |  dr  |
          |                   |     |      |      |      |
          |  dx    dy    dz   |     | dN_k |      | dN_k |
          |  --    --    --   |  *  | ---- |   =  | ---- | for all k
          |  ds    ds    ds   |     |  dy  |      |  ds  |
          |                   |     |      |      |      |
          |  dx    dy    dz   |     | dN_k |      | dN_k |
          |  --    --    --   |     | ---- |      | ---- |
          |  dt    dt    dt   |     |  dz  |      |  dt  |
          +-                 -+     +-    -+      +-    -+

  */
  LINALG::Matrix<3,3> xji;
  xji(0,0) = (  xjm(1,1)*xjm(2,2) - xjm(2,1)*xjm(1,2))/det;
  xji(1,0) = (- xjm(1,0)*xjm(2,2) + xjm(2,0)*xjm(1,2))/det;
  xji(2,0) = (  xjm(1,0)*xjm(2,1) - xjm(2,0)*xjm(1,1))/det;
  xji(0,1) = (- xjm(0,1)*xjm(2,2) + xjm(2,1)*xjm(0,2))/det;
  xji(1,1) = (  xjm(0,0)*xjm(2,2) - xjm(2,0)*xjm(0,2))/det;
  xji(2,1) = (- xjm(0,0)*xjm(2,1) + xjm(2,0)*xjm(0,1))/det;
  xji(0,2) = (  xjm(0,1)*xjm(1,2) - xjm(1,1)*xjm(0,2))/det;
  xji(1,2) = (- xjm(0,0)*xjm(1,2) + xjm(1,0)*xjm(0,2))/det;
  xji(2,2) = (  xjm(0,0)*xjm(1,1) - xjm(1,0)*xjm(0,1))/det;

  // compute global derivates
  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<iel;++rr)
    {
      derxy(nn,rr)=deriv(0,rr)*xji(nn,0);
      for (int mm=1;mm<3;++mm)
      {
        derxy(nn,rr)+=deriv(mm,rr)*xji(nn,mm);
      }
    }
  }

  // get velocities (n+alpha_F/1,i) at integration point
  //
  //                   +-----
  //     ^ n+af/1       \                ^ n+af/1
  //    vel      (x) =   +      N (x) * vel
  //                     /        j         j
  //                    +-----
  //                    node j
  //
  LINALG::Matrix<3,1> velint_hat;
  for (int rr=0;rr<3;++rr)
  {
    velint_hat(rr)=funct(0)*evel_hat(rr,0);
    for (int mm=1;mm<iel;++mm)
    {
      velint_hat(rr)+=funct(mm)*evel_hat(rr,mm);
    }
  }

  // get velocity (n+alpha_F,i) derivatives at integration point
  //
  //     ^ n+af/1      +-----  dN (x)
  //   dvel      (x)    \        k       ^ n+af/1
  //   ------------- =   +     ------ * vel
  //       dx           /        dx        k
  //         j         +-----      j
  //                   node k
  //
  // j : direction of derivative x/y/z
  //
  LINALG::Matrix<3,3>  vderxy_hat;

  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<3;++rr)
    {
      vderxy_hat(nn,rr)=derxy(rr,0)*evel_hat(nn,0);
      for (int mm=1;mm<iel;++mm)
      {
        vderxy_hat(nn,rr)+=derxy(rr,mm)*evel_hat(nn,mm);
      }
    }
  }

  // get filtered reynolds stress (n+alpha_F/1,i) at integration point
  //
  //                        +-----
  //        ^   n+af/1       \                   ^   n+af/1
  //    restress      (x) =   +      N (x) * restress
  //            ij           /        k              k, ij
  //                        +-----
  //                        node k
  //
  LINALG::Matrix<3,3> restress_hat;

  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<3;++rr)
    {
      int index = 3*nn+rr;
      restress_hat(nn,rr)=funct(0)*ereynoldsstress_hat(index,0);

      for (int mm=1;mm<iel;++mm)
      {
        restress_hat(nn,rr)+=funct(mm)*ereynoldsstress_hat(index,mm);
      }
    }
  }

  // get filtered modeled subgrid stress (n+alpha_F/1,i) at integration point
  //
  //
  //                   ^                   n+af/1
  //    filtered_modeled_subgrid_stress_hat      (x) =
  //                                       ij
  //
  //            +-----
  //             \                              ^                   n+af/1
  //          =   +      N (x) * filtered_modeled_subgrid_stress_hat
  //             /        k                                         k, ij
  //            +-----
  //            node k
  //
  //
  LINALG::Matrix<3,3> filtered_modeled_subgrid_stress_hat;
  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<3;++rr)
    {
      int index = 3*nn+rr;
      filtered_modeled_subgrid_stress_hat(nn,rr)=funct(0)*efiltered_modeled_subgrid_stress_hat(index,0);

      for (int mm=1;mm<iel;++mm)
      {
        filtered_modeled_subgrid_stress_hat(nn,rr)+=funct(mm)*efiltered_modeled_subgrid_stress_hat(index,mm);
      }
    }
  }


  /*
                            +-   ^ n+af/1        ^   n+af/1    -+
      ^   / h \       1.0   |  dvel_i    (x)   dvel_j      (x)  |
     eps | u   |    = --- * |  ------------- + ---------------  |
          \   / ij    2.0   |       dx              dx          |
                            +-        j               i        -+
  */

  LINALG::Matrix<3,3>  epsilon_hat;
  for (int nn=0;nn<3;++nn)
  {
    for (int rr=0;rr<3;++rr)
    {
      epsilon_hat(nn,rr) = 0.5 * ( vderxy_hat(nn,rr) + vderxy_hat(rr,nn) );
    }
  }

  //
  // modeled part of subtestfilter scale stresses
  //
  /*    +-                                 -+ 1
        |      ^   / h \       ^   / h \    | -     ^   / h \
        | 2 * eps | u   |   * eps | u   |   | 2  * eps | u   |
        |          \   / kl        \   / kl |           \   / ij
        +-                                 -+

        |                                   |
        +-----------------------------------+
             'resolved' rate of strain
  */

  double rateofstrain_hat = 0;

  for(int rr=0;rr<3;rr++)
  {
    for(int mm=0;mm<3;mm++)
    {
      rateofstrain_hat += epsilon_hat(rr,mm)*epsilon_hat(rr,mm);
    }
  }
  rateofstrain_hat *= 2.0;
  rateofstrain_hat = sqrt(rateofstrain_hat);

  LINALG::Matrix<3,3> L_ij;
  LINALG::Matrix<3,3> M_ij;

  for(int rr=0;rr<3;rr++)
  {
    for(int mm=0;mm<3;mm++)
    {
      L_ij(rr,mm) = restress_hat(rr,mm) - velint_hat(rr)*velint_hat(mm);
    }
  }

  // this is sqrt(3)
  double filterwidthratio = 1.73;

  for(int rr=0;rr<3;rr++)
  {
    for(int mm=0;mm<3;mm++)
    {
      M_ij(rr,mm) = filtered_modeled_subgrid_stress_hat(rr,mm)
        -
        filterwidthratio*filterwidthratio*rateofstrain_hat*epsilon_hat(rr,mm);
    }
  }

  LijMij =0;
  MijMij =0;
  for(int rr=0;rr<3;rr++)
  {
    for(int mm=0;mm<3;mm++)
    {
      LijMij += L_ij(rr,mm)*M_ij(rr,mm);
      MijMij += M_ij(rr,mm)*M_ij(rr,mm);
    }
  }

  return;
} // DRT::ELEMENTS::Fluid::f3_calc_smag_const_LijMij_and_MijMij


//----------------------------------------------------------------------
//                                                       rasthofer 05/11
//----------------------------------------------------------------------
template<int NEN, int NSD, DRT::Element::DiscretizationType DISTYPE>
void DRT::ELEMENTS::Fluid::f3_get_mf_params(
  Teuchos::ParameterList&      params,
  RCP<MAT::Material>  mat,
  vector<double>&     vel,
  vector<double>&     fsvel)
{
  // get mfs parameter
	Teuchos::ParameterList * turbmodelparamsmfs = &(params.sublist("MULTIFRACTAL SUBGRID SCALES"));
  bool withscatra = params.get<bool>("scalar");

  // allocate a fixed size array for nodal velocities
  LINALG::Matrix<NSD,NEN>   evel;
  LINALG::Matrix<NSD,NEN>   efsvel;

  // split velocity and throw away  pressure, insert into element array
  for (int inode=0;inode<NEN;inode++)
  {
    for (int idim=0;idim<NSD;idim++)
    {
      evel(idim,inode) = vel[inode*4+idim];
      efsvel(idim,inode) = fsvel[inode*4+idim];
    }
  }

  // get material
  double dynvisc = 0.0;
  double dens = 0.0;
  if (mat->MaterialType() == INPAR::MAT::m_fluid)
  {
    const MAT::NewtonianFluid* actmat = static_cast<const MAT::NewtonianFluid*>(mat.get());
    // get constant viscosity
    dynvisc = actmat->Viscosity();
    // get constant density
    dens = actmat->Density();
    if (dens == 0.0 or dynvisc == 0.0)
    {
      dserror("Could not get material parameters!");
    }
  }

  // allocate array for gauss-point velocities and derivatives
  LINALG::Matrix<NSD,1> velint;
  LINALG::Matrix<NSD,NSD> velintderxy;
  LINALG::Matrix<NSD,1> fsvelint;
  LINALG::Matrix<NSD,NSD> fsvelintderxy;

  // allocate arrays for shape functions and derivatives
  LINALG::Matrix<NEN,1> funct;
  LINALG::Matrix<NSD,NEN> deriv;
  LINALG::Matrix<NSD,NEN> derxy;
  double vol = 0.0;

  // array for element coordinates in physical space
  LINALG::Matrix<NSD,NEN> xyze;
  //this will be the y-coordinate of the element center
  double center = 0;
  // get node coordinates of element
  for(int inode=0;inode<NEN;inode++)
  {
    for (int idim=0;idim<NSD;idim++)
      xyze(idim,inode)=this->Nodes()[inode]->X()[idim];

    center+=xyze(1,inode);
  }
  center/=NEN;

  // evaluate shape functions and derivatives at element center
  LINALG::Matrix<NSD,NSD> xji;
  {
    // use one-point Gauss rule
    DRT::UTILS::IntPointsAndWeights<NSD> intpoints(DRT::ELEMENTS::DisTypeToStabGaussRule<DISTYPE>::rule);

    // coordinates of the current integration point
    const double* gpcoord = (intpoints.IP().qxg)[0];
    LINALG::Matrix<NSD,1> xsi;
    for (int idim=0;idim<NSD;idim++)
    {
      xsi(idim) = gpcoord[idim];
    }
    const double wquad = intpoints.IP().qwgt[0];

    // shape functions and their first derivatives
    DRT::UTILS::shape_function<DISTYPE>(xsi,funct);
    DRT::UTILS::shape_function_deriv1<DISTYPE>(xsi,deriv);

    // get Jacobian matrix and determinant
    LINALG::Matrix<NSD,NSD> xjm;
    //LINALG::Matrix<NSD,NSD> xji;
    xjm.MultiplyNT(deriv,xyze);
    double det = xji.Invert(xjm);
    // check for degenerated elements
    if (det < 1E-16)
      dserror("GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", this->Id(), det);

    // set element area or volume
    vol = wquad*det;

    // compute global first derivatives
    derxy.Multiply(xji,deriv);
  }

  // calculate parameters of multifractal subgrid-scales
  // set input parameters
  double Csgs = turbmodelparamsmfs->get<double>("CSGS");
  double alpha = 0.0;
  if (turbmodelparamsmfs->get<string>("SCALE_SEPARATION") == "algebraic_multigrid_operator")
   alpha = 3.0;
  else if (turbmodelparamsmfs->get<string>("SCALE_SEPARATION") == "box_filter"
        or turbmodelparamsmfs->get<string>("SCALE_SEPARATION") == "geometric_multigrid_operator")
   alpha = 2.0;
  else
   dserror("Unknown filter type!");
  // allocate vector for parameter N
  // N may depend on the direction
  vector<double> Nvel (3);
  // element Reynolds number
  double Re_ele = -1.0;
  // characteristic element length
  double hk = 1.0e+10;

  // calculate norm of strain rate
  double strainnorm = 0.0;
  // compute (resolved) norm of strain
  //
  //          +-                                 -+ 1
  //          |          /   \           /   \    | -
  //          |     eps | vel |   * eps | vel |   | 2
  //          |          \   / ij        \   / ij |
  //          +-                                 -+
  //
  velintderxy.MultiplyNT(evel,derxy);
  LINALG::Matrix<NSD,NSD> twoeps;
  for(int idim=0;idim<NSD;idim++)
  {
    for(int jdim=0;jdim<NSD;jdim++)
    {
      twoeps(idim,jdim) = velintderxy(idim,jdim) + velintderxy(jdim,idim);
    }
  }

  for(int idim=0;idim<NSD;idim++)
  {
    for(int jdim=0;jdim<NSD;jdim++)
    {
      strainnorm += twoeps(idim,jdim)*twoeps(idim,jdim);
    }
  }
  strainnorm = (sqrt(strainnorm/4.0));

  // do we have a fixed parameter N
  if ((DRT::INPUT::IntegralValue<int>(*turbmodelparamsmfs,"CALC_N")) == false)
  {
    for (int rr=1;rr<3;rr++)
      Nvel[rr] = turbmodelparamsmfs->get<double>("N");
#ifdef DIR_N // direction dependent stuff, currently not used
  Nvel[0] = NUMX;
  Nvel[1] = NUMY;
  Nvel[2] = NUMZ;
#endif
  }
  else //no, so we calculate N from Re
  {
  double scale_ratio = 0.0;

  // get velocity at element center
  velint.Multiply(evel,funct);
  fsvelint.Multiply(efsvel,funct);
  // get norm
  const double vel_norm = velint.Norm2();
  const double fsvel_norm = fsvelint.Norm2();

  // calculate characteristic element length
  // cf. stabilization parameters
  INPAR::FLUID::RefLength reflength = INPAR::FLUID::cube_edge;
  if (turbmodelparamsmfs->get<string>("REF_LENGTH") == "cube_edge")
   reflength = INPAR::FLUID::cube_edge;
  else if (turbmodelparamsmfs->get<string>("REF_LENGTH") == "sphere_diameter")
   reflength = INPAR::FLUID::sphere_diameter;
  else if (turbmodelparamsmfs->get<string>("REF_LENGTH") == "streamlength")
   reflength = INPAR::FLUID::streamlength;
  else if (turbmodelparamsmfs->get<string>("REF_LENGTH") == "gradient_based")
   reflength = INPAR::FLUID::gradient_based;
  else if (turbmodelparamsmfs->get<string>("REF_LENGTH") == "metric_tensor")
   reflength = INPAR::FLUID::metric_tensor;
  else
   dserror("Unknown length!");
  switch (reflength)
  {
    case INPAR::FLUID::streamlength:
    {
        // a) streamlength due to Tezduyar et al. (1992)
        // normed velocity vector
        LINALG::Matrix<NSD,1> velino(true);
        if (vel_norm>=1e-6) velino.Update(1.0/vel_norm,velint);
        else
        {
          velino.Clear();
          velino(0,0) = 1.0;
        }
        LINALG::Matrix<NEN,1> tmp;
        tmp.MultiplyTN(derxy,velino);
        const double val = tmp.Norm1();
        hk = 2.0/val;

      break;
    }
    case INPAR::FLUID::sphere_diameter:
    {
      // b) volume-equivalent diameter
      hk = pow((6.*vol/M_PI),(1.0/3.0))/sqrt(3.0);

      break;
    }
    case INPAR::FLUID::cube_edge:
    {
      // c) qubic element length
      hk = pow(vol,(1.0/NSD));
      break;
    }
    case INPAR::FLUID::metric_tensor:
    {
        /*          +-           -+   +-           -+   +-           -+
                    |             |   |             |   |             |
                    |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
              G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
               ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                    |    i     j  |   |    i     j  |   |    i     j  |
                    +-           -+   +-           -+   +-           -+
        */
        LINALG::Matrix<3,3> G;

        for (int nn=0;nn<3;++nn)
        {
          for (int rr=0;rr<3;++rr)
          {
            G(nn,rr) = xji(nn,0)*xji(rr,0);
            for (int mm=1;mm<3;++mm)
            {
              G(nn,rr) += xji(nn,mm)*xji(rr,mm);
            }
          }
        }

        /*          +----
                     \
            G : G =   +   G   * G
            -   -    /     ij    ij
            -   -   +----
                     i,j
        */
        double normG = 0;
        for (int nn=0;nn<3;++nn)
        {
          for (int rr=0;rr<3;++rr)
          {
            normG+=G(nn,rr)*G(nn,rr);
          }
        }
        hk = pow(normG,-0.25);

      break;
    }
    case INPAR::FLUID::gradient_based:
    {
       velintderxy.MultiplyNT(evel,derxy);
       LINALG::Matrix<3,1> normed_velgrad;

       for (int rr=0;rr<3;++rr)
       {
         normed_velgrad(rr)=sqrt(velintderxy(0,rr)*velintderxy(0,rr)
                               + velintderxy(1,rr)*velintderxy(1,rr)
                               + velintderxy(2,rr)*velintderxy(2,rr));
       }
       double norm=normed_velgrad.Norm2();

       // normed gradient
       if (norm>1e-6)
       {
         for (int rr=0;rr<3;++rr)
         {
           normed_velgrad(rr)/=norm;
         }
       }
       else
       {
         normed_velgrad(0) = 1.;
         for (int rr=1;rr<3;++rr)
         {
           normed_velgrad(rr)=0.0;
         }
       }

       // get length in this direction
       double val = 0.0;
       for (int rr=0;rr<NEN;++rr) /* loop element nodes */
       {
         val += fabs( normed_velgrad(0)*derxy(0,rr)
                     +normed_velgrad(1)*derxy(1,rr)
                     +normed_velgrad(2)*derxy(2,rr));
       } /* end of loop over element nodes */

       hk = 2.0/val;

      break;
    }
    default:
      dserror("Unknown length");
  }

  // alternative length for comparison, currently not used
#ifdef HMIN
      double xmin = 0.0;
      double ymin = 0.0;
      double zmin = 0.0;
      double xmax = 0.0;
      double ymax = 0.0;
      double zmax = 0.0;
      for (int inen=0; inen<NEN; inen++)
      {
        if (inen == 0)
        {
          xmin = xyze(0,inen);
          xmax = xyze(0,inen);
          ymin = xyze(1,inen);
          ymax = xyze(1,inen);
          zmin = xyze(2,inen);
          zmax = xyze(2,inen);
        }
        else
        {
          if(xyze(0,inen)<xmin)
            xmin = xyze(0,inen);
          if(xyze(0,inen)>xmax)
            xmax = xyze(0,inen);
          if(xyze(1,inen)<ymin)
            ymin = xyze(1,inen);
          if(xyze(1,inen)>ymax)
            ymax = xyze(1,inen);
          if(xyze(2,inen)<zmin)
            zmin = xyze(2,inen);
          if(xyze(2,inen)>zmax)
            zmax = xyze(2,inen);
        }
      }
      if ((xmax-xmin) < (ymax-ymin))
      {
        if ((xmax-xmin) < (zmax-zmin))
           hk = xmax-xmin;
      }
      else
      {
        if ((ymax-ymin) < (zmax-zmin))
           hk = ymax-ymin;
        else
           hk = zmax-zmin;
      }
#endif
#ifdef HMAX
      double xmin = 0.0;
      double ymin = 0.0;
      double zmin = 0.0;
      double xmax = 0.0;
      double ymax = 0.0;
      double zmax = 0.0;
      for (int inen=0; inen<NEN; inen++)
      {
        if (inen == 0)
        {
          xmin = xyze(0,inen);
          xmax = xyze(0,inen);
          ymin = xyze(1,inen);
          ymax = xyze(1,inen);
          zmin = xyze(2,inen);
          zmax = xyze(2,inen);
        }
        else
        {
          if(xyze(0,inen)<xmin)
            xmin = xyze(0,inen);
          if(xyze(0,inen)>xmax)
            xmax = xyze(0,inen);
          if(xyze(1,inen)<ymin)
            ymin = xyze(1,inen);
          if(xyze(1,inen)>ymax)
            ymax = xyze(1,inen);
          if(xyze(2,inen)<zmin)
            zmin = xyze(2,inen);
          if(xyze(2,inen)>zmax)
            zmax = xyze(2,inen);
        }
      }
      if ((xmax-xmin) > (ymax-ymin))
      {
        if ((xmax-xmin) > (zmax-zmin))
           hk = xmax-xmin;
      }
      else
      {
        if ((ymax-ymin) > (zmax-zmin))
           hk = ymax-ymin;
        else
           hk = zmax-zmin;
      }
#endif

  if (hk == 1.0e+10)
    dserror("Something went wrong!");

  // get reference velocity
  INPAR::FLUID::RefVelocity refvel = INPAR::FLUID::strainrate;
  if (turbmodelparamsmfs->get<string>("REF_VELOCITY") == "strainrate")
   refvel = INPAR::FLUID::strainrate;
  else if (turbmodelparamsmfs->get<string>("REF_VELOCITY") == "resolved")
   refvel = INPAR::FLUID::resolved;
  else if (turbmodelparamsmfs->get<string>("REF_VELOCITY") == "fine_scale")
   refvel = INPAR::FLUID::fine_scale;
  else
   dserror("Unknown velocity!");

  switch (refvel){
  case INPAR::FLUID::resolved:
  {
    Re_ele = vel_norm * hk *dens / dynvisc;
    break;
  }
  case INPAR::FLUID::fine_scale:
  {
    Re_ele = fsvel_norm * hk *dens / dynvisc;
    break;
  }
  case INPAR::FLUID::strainrate:
  {
    Re_ele = strainnorm * hk * hk * dens / dynvisc;
    break;
  }
  default:
    dserror("Unknown velocity!");
  }
  if (Re_ele < 0.0)
    dserror("Something went wrong!");

  if (Re_ele < 1.0)
     Re_ele = 1.0;

  //
  //   Delta
  //  ---------  ~ Re^(3/4)
  //  lambda_nu
  //
  scale_ratio = pow(Re_ele,3.0/4.0);
  scale_ratio = turbmodelparamsmfs->get<double>("C_NU") * pow(Re_ele,3.0/4.0);
  // scale_ration < 1.0 leads to N < 0
  // therefore, we clip once more
  if (scale_ratio < 1.0)
    scale_ratio = 1.0;

  //         |   Delta     |
  //  N =log | ----------- |
  //        2|  lambda_nu  |
  double N_re = log(scale_ratio)/log(2.0);
  if (N_re < 0.0)
    dserror("Something went wrong when calculating N!");

  for (int i=0; i<NSD; i++)
    Nvel[i] = N_re;
  }
#ifdef DIR_N
    vector<double> weights (3);
    weights[0] = WEIGHT_NX;
    weights[1] = WEIGHT_NY;
    weights[2] = WEIGHT_NZ;
    for (int i=0; i<NSD; i++)
      Nvel[i] *= weights[i];
#endif


  // calculate coefficient of subgrid-velocity
  // allocate array for coefficient B
  // B may depend on the direction (if N depends on it)
  LINALG::Matrix<NSD,1> B(true);
  {
    //                                  1
    //          |       1              |2
    //  kappa = | -------------------- |
    //          |  1 - alpha ^ (-4/3)  |
    //
    double kappa = 1.0/(1.0-pow(alpha,-4.0/3.0));

    //                                                     1
    //                                  |                 |2
    //  B = CI * kappa * 2 ^ (-2*N/3) * | 2 ^ (4*N/3) - 1 |
    //                                  |                 |
    //


    // calculate near-wall correction
    if ((DRT::INPUT::IntegralValue<int>(*turbmodelparamsmfs,"NEAR_WALL_LIMIT")) == true)
    {
      // get Re from strain rate
      double Re_ele_str = strainnorm * hk * hk * dens / dynvisc;
      if (Re_ele_str < 0.0)
        dserror("Something went wrong!");
      // ensure positive values
      if (Re_ele_str < 1.0)
        Re_ele_str = 1.0;

      // calculate corrected Csgs
      //           -3/16
      //  *(1 - (Re)   )
      //
      Csgs *= (1-pow(Re_ele_str,-3.0/16.0));
    }

    for (int dim=0; dim<NSD; dim++)
    {
      B(dim,0) = Csgs * sqrt(kappa) * pow(2.0,-2.0*Nvel[0]/3.0) * sqrt((pow(2.0,4.0*Nvel[0]/3.0)-1));
    }
  }

#ifdef CONST_B
  for (int dim=0; dim<NSD; dim++)
  {
    B(dim,0) = B_CONST;
  }
#endif

  // calculate model parameters for passive scalar transport
  // allocate vector for parameter N
  // N may depend on the direction -> currently unused
  double Nphi= 0.0;
  // allocate array for coefficient D
  // D may depend on the direction (if N depends on it)
  double D = 0.0;
  double Csgs_phi = turbmodelparamsmfs->get<double>("CSGS_PHI");
  if (withscatra)
  {
    // get Schmidt number
    double scnum = params.get<double>("scnum");
    // ratio of dissipation scale to element length
    double scale_ratio_phi = 0.0;

    if ((DRT::INPUT::IntegralValue<int>(*turbmodelparamsmfs,"CALC_N")) == true)
    {
      //
      //   Delta
      //  ---------  ~ Re^(3/4)*Sc^(1/2)
      //  lambda_diff
      //
      scale_ratio_phi = turbmodelparamsmfs->get<double>("C_DIFF") * pow(Re_ele,3.0/4.0) * pow(scnum,1.0/2.0);
      // scale_ratio < 1.0 leads to N < 0
      // therefore, we clip again
      if (scale_ratio_phi < 1.0)
        scale_ratio_phi = 1.0;

      //         |   Delta     |
      //  N =log | ----------- |
      //        2|  lambda_nu  |
      Nphi = log(scale_ratio_phi)/log(2.0);
      if (Nphi < 0.0)
        dserror("Something went wrong when calculating N!");
    }
    else
     dserror("Multifractal subgrid-scales for scalar transport with calculation of N, only!");

    // here, we have to distinguish three different cases:
    // Sc ~ 1 : fluid and scalar field have the nearly the same cutoff (usual case)
    //          k^(-5/3) scaling -> gamma = 4/3
    // Sc >> 1: (i)  cutoff in the inertial-convective range (Nvel>0, tricky!)
    //               k^(-5/3) scaling in the inertial-convective range
    //               k^(-1) scaling in the viscous-convective range
    //          (ii) cutoff in the viscous-convective range (fluid field fully resolved, easier)
    //               k^(-1) scaling -> gamma = 2
    // rare:
    // Sc << 1: fluid field could be fully resolved, not necessary
    //          k^(-5/3) scaling -> gamma = 4/3
    // Remark: case 2.(i) not implemented, yet

#ifndef TESTING
    double gamma = 0.0;
    if (scnum < 2.0) // Sc <= 1, i.e., case 1 and 3
      gamma = 4.0/3.0;
    else if (scnum > 2.0 and Nvel[0]<1.0) // Pr >> 1, i.e., case 2 (ii)
      gamma = 2.0;
    else if (scnum > 2.0 and Nvel[0]<Nphi)
      dserror("Inertial-convective and viscous-convective range?");
    else
      dserror("Could not determine gamma!");

    //
    //   Phi    |       1                |
    //  kappa = | ---------------------- |
    //          |  1 - alpha ^ (-gamma)  |
    //
    double kappa_phi = 1.0/(1.0-pow(alpha,-gamma));

    // calculate coefficient of subgrid-scalar
    //                                                             1
    //       Phi    Phi                       |                   |2
    //  D = Csgs * kappa * 2 ^ (-gamma*N/2) * | 2 ^ (gamma*N) - 1 |
    //                                        |                   |
    //
    D = Csgs_phi *sqrt(kappa_phi) * pow(2.0,-gamma*Nphi/2.0) * sqrt((pow(2.0,gamma*Nphi)-1));
#endif

    // second implementation for testing on cluster
#ifdef TESTING
  double fac = 1.0;
# if 1
    if ((DRT::INPUT::IntegralValue<int>(*turbmodelparamsmfs,"NEAR_WALL_LIMIT")) == true)
    {
      // get Re from strain rate
      double Re_ele_str = strainnorm * hk * hk * dens / dynvisc;
      if (Re_ele_str < 0.0)
        dserror("Something went wrong!");
      // ensure positive values
      if (Re_ele_str < 1.0)
        Re_ele_str = 1.0;

      // calculate corrected Csgs
      //           -3/16
      //  *(1 - (Re)   )
      //
      double Pr = params.get<double>("scnum");
      fac = (1-pow(Re_ele_str,-3.0/16.0)); //*pow(Pr,-1.0/8.0));
    }
#endif


  // Pr <= 1
  # if 1
    double gamma = 0.0;
    gamma = 4.0/3.0;
    double kappa_phi = 1.0/(1.0-pow(alpha,-gamma));
    D = Csgs_phi *sqrt(kappa_phi) * pow(2.0,-gamma*Nphi/2.0) * sqrt((pow(2.0,gamma*Nphi)-1))*fac;
  #endif

  // Pr >> 1: cutoff viscous-convective
  # if 0
    double gamma = 0.0;
    gamma = 2.0;
    double kappa_phi = 1.0/(1.0-pow(alpha,-gamma));
    D(dim,0) = Csgs_phi *sqrt(kappa_phi) * pow(2.0,-gamma*Nphi/2.0) * sqrt((pow(2.0,gamma*Nphi)-1))*fac;
  #endif

  // Pr >> 1: cutoff inertial-convective
  #if 0
    double gamma1 = 0.0;
    gamma1 = 4.0/3.0;
    double gamma2 = 0.0;
    gamma2 = 2.0;
    double kappa_phi = 1.0/(1.0-pow(alpha,-gamma1));
      D = Csgs_phi * sqrt(kappa_phi) * pow(2.0,-gamma2*Nphi/2.0) * sqrt((pow(2.0,gamma1*Nvel[dim])-1)+4.0/3.0*(PI/hk)*(pow(2.0,gamma2*Nphi)-pow(2.0,gamma2*Nvel[dim])))*fac;
  #endif
#endif


  }

  // calculate subgrid-viscosity, if small-scale eddy-viscosity term is included
  double sgvisc = 0.0;
  if (params.sublist("TURBULENCE MODEL").get<string>("FSSUGRVISC","No") != "No")
  {
    // get filter width and Smagorinsky-coefficient
    const double hk_sgvisc = pow(vol,(1.0/NSD));
    const double Cs = params.sublist("SUBGRID VISCOSITY").get<double>("C_SMAGORINSKY");

    // compute rate of strain
    //
    //          +-                                 -+ 1
    //          |          /   \           /   \    | -
    //          | 2 * eps | vel |   * eps | vel |   | 2
    //          |          \   / ij        \   / ij |
    //          +-                                 -+
    //
    LINALG::Matrix<NSD,NSD> velderxy(true);
    velintderxy.MultiplyNT(evel,derxy);
    fsvelintderxy.MultiplyNT(efsvel,derxy);

    if (params.sublist("TURBULENCE MODEL").get<string>("FSSUGRVISC","No") == "Smagorinsky_all")
      velderxy = velintderxy;
    else if (params.sublist("TURBULENCE MODEL").get<string>("FSSUGRVISC","No") == "Smagorinsky_small")
      velderxy = fsvelintderxy;
    else
      dserror("fssgvisc-type unknown");

#ifdef SUBGRID_SCALE //unused
    for (int idim=0; idim<NSD; idim++)
    {
      for (int jdim=0; jdim<NSD; jdim++)
        mffsvelintderxy_(idim,jdim) = fsvelintderxy_(idim,jdim) * B(idim,0);
    }
    velderxy = mffsvelintderxy;
#endif
    LINALG::Matrix<NSD,NSD> two_epsilon;
    double rateofstrain = 0.0;
    for (int idim=0; idim<NSD; idim++)
    {
      for (int jdim=0; jdim<NSD; jdim++)
      {
        two_epsilon(idim,jdim) = velderxy(idim,jdim) + velderxy(jdim,idim);
      }
    }

    for (int idim=0; idim<NSD; idim++)
    {
      for (int jdim=0; jdim<NSD; jdim++)
      {
        rateofstrain += two_epsilon(idim,jdim)*two_epsilon(jdim,idim);
      }
    }

    rateofstrain = (sqrt(rateofstrain/2.0));

    //                                      +-                                 -+ 1
    //                                  2   |          /    \          /   \    | -
    //    visc          = dens * (C_S*h)  * | 2 * eps | vel |   * eps | vel |   | 2
    //        turbulent                     |          \   / ij        \   / ij |
    //                                      +-                                 -+
    //                                      |                                   |
    //                                      +-----------------------------------+
    //                                                   rate of strain
    sgvisc = dens * Cs * Cs * hk_sgvisc * hk_sgvisc * rateofstrain;

  }

  // set parameter in sublist turbulence
  Teuchos::ParameterList * modelparams =&(params.sublist("TURBULENCE MODEL"));
  RCP<vector<double> > sum_N_stream      = modelparams->get<RCP<vector<double> > >("local_N_stream_sum");
  RCP<vector<double> > sum_N_normal      = modelparams->get<RCP<vector<double> > >("local_N_normal_sum");
  RCP<vector<double> > sum_N_span        = modelparams->get<RCP<vector<double> > >("local_N_span_sum");
  RCP<vector<double> > sum_B_stream      = modelparams->get<RCP<vector<double> > >("local_B_stream_sum");
  RCP<vector<double> > sum_B_normal      = modelparams->get<RCP<vector<double> > >("local_B_normal_sum");
  RCP<vector<double> > sum_B_span        = modelparams->get<RCP<vector<double> > >("local_B_span_sum");
  RCP<vector<double> > sum_Csgs          = modelparams->get<RCP<vector<double> > >("local_Csgs_sum");
  RCP<vector<double> > sum_Nphi;
  RCP<vector<double> > sum_Dphi;
  RCP<vector<double> > sum_Csgs_phi;
  if (withscatra)
  {
    sum_Nphi          = modelparams->get<RCP<vector<double> > >("local_Nphi_sum");
    sum_Dphi          = modelparams->get<RCP<vector<double> > >("local_Dphi_sum");
    sum_Csgs_phi      = modelparams->get<RCP<vector<double> > >("local_Csgs_phi_sum");
  }
  RCP<vector<double> > sum_sgvisc        = modelparams->get<RCP<vector<double> > >("local_sgvisc_sum");

  // the coordinates of the element layers in the channel
  // planecoords are named nodeplanes in turbulence_statistics_channel!
  RCP<vector<double> > planecoords  = modelparams->get<RCP<vector<double> > >("planecoords",Teuchos::null);
  if(planecoords==Teuchos::null)
    dserror("planecoords is null, but need channel_flow_of_height_2\n");

  bool found = false;
  int nlayer = 0;
  for (nlayer=0;nlayer<(int)(*planecoords).size()-1;)
  {
    if(center<(*planecoords)[nlayer+1])
    {
      found = true;
      break;
    }
    nlayer++;
  }
  if (found ==false)
  {
    dserror("could not determine element layer");
  }

  (*sum_N_stream)[nlayer] += Nvel[0];
  (*sum_N_normal)[nlayer] += Nvel[1];
  (*sum_N_span)[nlayer] += Nvel[2];
  (*sum_B_stream)[nlayer] += B(0,0);
  (*sum_B_normal)[nlayer] += B(1,0);
  (*sum_B_span)[nlayer] += B(2,0);
  (*sum_Csgs)[nlayer] += Csgs;
  if (withscatra)
  {
    (*sum_Csgs_phi)[nlayer] += Csgs_phi;
    (*sum_Nphi)[nlayer] += Nphi;
    (*sum_Dphi)[nlayer] += D;
  }
  (*sum_sgvisc)[nlayer] += sgvisc;

  return;
} // DRT::ELEMENTS::Fluid::f3_get_mf_params


//----------------------------------------------------------------------
//----------------------------------------------------------------------
template<DRT::Element::DiscretizationType DISTYPE>
void DRT::ELEMENTS::Fluid::ElementNodeNormal(Teuchos::ParameterList&     params,
                                              DRT::Discretization&       discretization,
                                              vector<int>&               lm,
                                              Epetra_SerialDenseVector&  elevec1)
{
  // this evaluates the node normals using the volume integral in Wall
  // (7.13). That formula considers all surfaces of the element, not only the
  // free surfaces. This causes difficulties because the free surface normals
  // point outwards on nodes at the rim of a basin (e.g. channel-flow).

  // get number of nodes
  const int iel = DRT::UTILS::DisTypeToNumNodePerEle<DISTYPE>::numNodePerElement;

  // get number of dimensions
  const int nsd = DRT::UTILS::DisTypeToDim<DISTYPE>::dim;

  // get number of dof's
  const int numdofpernode = nsd +1;


  /*
  // create matrix objects for nodal values
  LINALG::Matrix<3,iel>       edispnp;

  if (is_ale_)
  {
    // get most recent displacements
    RCP<const Epetra_Vector> dispnp
      =
      discretization.GetState("dispnp");

    if (dispnp==null)
    {
      dserror("Cannot get state vector 'dispnp'");
    }

    vector<double> mydispnp(lm.size());
    DRT::UTILS::ExtractMyValues(*dispnp,mydispnp,lm);

    // extract velocity part from "mygridvelaf" and get
    // set element displacements
    for (int i=0;i<iel;++i)
    {
      int fi    =4*i;
      int fip   =fi+1;
      int fipp  =fip+1;
      edispnp(0,i)    = mydispnp   [fi  ];
      edispnp(1,i)    = mydispnp   [fip ];
      edispnp(2,i)    = mydispnp   [fipp];
    }
  }

  // set element data
  const DiscretizationType distype = this->Shape();
*/
  //----------------------------------------------------------------------------
  //                         ELEMENT GEOMETRY
  //----------------------------------------------------------------------------
  //LINALG::Matrix<3,iel>  xyze;
  LINALG::Matrix<nsd,iel>  xyze;

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary element!)
  GEO::fillInitialPositionArray<DISTYPE,nsd, LINALG::Matrix<nsd,iel> >(this,xyze);

  /*
  // get node coordinates
  DRT::Node** nodes = Nodes();
  for (int inode=0; inode<iel; inode++)
  {
    const double* x = nodes[inode]->X();
    xyze(0,inode) = x[0];
    xyze(1,inode) = x[1];
    xyze(2,inode) = x[2];
  }
*/
  if(is_ale_)
  {
    // --------------------------------------------------
    // create matrix objects for nodal values
    LINALG::Matrix<nsd,iel>       edispnp(true);

    // get most recent displacements
    RCP<const Epetra_Vector> dispnp
      =
      discretization.GetState("dispnp");

    if (dispnp==null)
    {
      dserror("Cannot get state vector 'dispnp'");
    }

    vector<double> mydispnp(lm.size());
    DRT::UTILS::ExtractMyValues(*dispnp,mydispnp,lm);

    // extract velocity part from "mygridvelaf" and get
    // set element displacements
    for (int inode=0; inode<iel; ++inode)
    {
      for(int idim=0; idim < nsd; ++idim)
      {
        edispnp(idim,inode)    = mydispnp   [numdofpernode*inode+idim ];
      }
    }
    // get new node positions for isale
    xyze += edispnp;
  }

  /*
  // add displacement, when fluid nodes move in the ALE case
  if (is_ale_)
  {
    for (int inode=0; inode<iel; inode++)
    {
      xyze(0,inode) += edispnp(0,inode);
      xyze(1,inode) += edispnp(1,inode);
      xyze(2,inode) += edispnp(2,inode);
    }
  }
*/
  //------------------------------------------------------------------
  //                       INTEGRATION LOOP
  //------------------------------------------------------------------
  //LINALG::Matrix<iel,1  > funct;
  //LINALG::Matrix<3,  iel> deriv;
  //LINALG::Matrix<3,  3  > xjm;
  //LINALG::Matrix<3,  3  > xji;

  LINALG::Matrix<iel,1>   funct(true);
  LINALG::Matrix<nsd,iel>   deriv(true);
  //LINALG::Matrix<6,6>   bm(true);

  // get Gaussrule
    const DRT::UTILS::IntPointsAndWeights<nsd> intpoints(DRT::ELEMENTS::DisTypeToOptGaussRule<DISTYPE>::rule);

  // gaussian points
  //const GaussRule3D          gaussrule = getOptimalGaussrule(distype);
  //const IntegrationPoints3D  intpoints(gaussrule);

  for (int iquad=0;iquad<intpoints.IP().nquad;++iquad)
  {
    // local Gauss point coordinates
    LINALG::Matrix<nsd,1>   xsi(true);

    // local coordinates of the current integration point
    const double* gpcoord = (intpoints.IP().qxg)[iquad];
    for (int idim=0;idim<nsd;++idim)
    {
      xsi(idim) = gpcoord[idim];
    }
/*
    // set gauss point coordinates
    LINALG::Matrix<3,1> gp;

    gp(0)=intpoints.qxg[iquad][0];
    gp(1)=intpoints.qxg[iquad][1];
    gp(2)=intpoints.qxg[iquad][2];

    if(!(distype == DRT::Element::nurbs8
         ||
         distype == DRT::Element::nurbs27))
    {
      // get values of shape functions and derivatives in the gausspoint
      DRT::UTILS::shape_function_3D       (funct,gp(0),gp(1),gp(2),distype);
      DRT::UTILS::shape_function_3D_deriv1(deriv,gp(0),gp(1),gp(2),distype);
    }
    else
    {
      dserror("not implemented");
    }
*/

    if(not IsNurbs<DISTYPE>::isnurbs)
    {
      // get values of shape functions and derivatives in the gausspoint
      //DRT::UTILS::shape_function_3D       (funct,gp(0),gp(1),gp(2),distype);
      //DRT::UTILS::shape_function_3D_deriv1(deriv,gp(0),gp(1),gp(2),distype);

      // shape function derivs of boundary element at gausspoint
      DRT::UTILS::shape_function<DISTYPE>(xsi,funct);
      DRT::UTILS::shape_function_deriv1<DISTYPE>(xsi,deriv);
    }
    else dserror("Nurbs are not implemented yet");

    LINALG::Matrix<nsd,nsd>   xjm(true);
    LINALG::Matrix<nsd,nsd>   xji(true);

    // compute jacobian matrix
    // determine jacobian at point r,s,t
    xjm.MultiplyNT(deriv,xyze);

    // determinant and inverse of jacobian
    const double det = xji.Invert(xjm);

    // check for degenerated elements
    if (det < 0.0)
    {
      dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", Id(), det);
    }

    // set total integration factor
    const double fac = intpoints.IP().qwgt[iquad] * det;

    // integrate shapefunction gradient over element
    for (int dim = 0; dim < 3; dim++)
    {
      for (int node = 0; node < iel; node++)
      {
        elevec1[4 * node + dim] += (deriv(0,node)*xji(dim,0) + deriv(1,node)*xji(dim,1) + deriv(2,node)*xji(dim,2))
                                   * fac;
      }
    }
  }

  return;
}

