/*----------------------------------------------------------------------*/
/*!
\file drt_function.cpp

\brief Managing and evaluating of spatial functions

<pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15238
</pre>
*/
/*----------------------------------------------------------------------*/



#include "drt_discret.H"
#include "drt_function.H"
#include "drt_globalproblem.H"
#include "standardtypes_cpp.H"
#include "drt_timecurve.H"
#include "drt_linedefinition.H"
#include "../drt_mat/newtonianfluid.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_io/io.H"

namespace DRT {
namespace UTILS {

  /// spatial function based on parsed expression
  class ExprFunction : public Function
  {
  public:
    /*!

    \brief construct spatial function from expression with given origin

    \note  Upon construction, the object defines a spatial function
           returning the same function value for every spatial dimension.
           If a vector-valued spatial function is required, further
           expressions can be added via the AddExpr function. In this
           case, the spatial function defined upon construction will be
           the first component of the vector-valued spatial function.

    \param buf (i) (c-string) expression to be parsed during evaluation
    \param x   (i) x-coordinate of the origin of the coordinate system
    \param y   (i) y-coordinate of the origin of the coordinate system
    \param z   (i) z-coordinate of the origin of the coordinate system

    */
    ExprFunction(char* buf, double x, double y, double z);

    /*!

    \brief Default constructor creating empty object. Expressions are
           added with add function

    */
    ExprFunction();

    /*!

    \brief clean up parse tree

    */
    ~ExprFunction();


    /*!

    \brief evaluate function at given position in space

    \param index (i) For vector-valued functions, index defines the
                     function-component which should be evaluated
                     For scalar functionsb, index is always set to 0
    \param x     (i) The point in 3-dimensional space in which the
                     function will be evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief add expression to an existing ExprFunction in order to extend
           it to a vector-valued spatial function.

           Every call to AddExpr adds one more component to the
           vector-valued function.

    \param buf (i) (c-string) expression to be parsed during evaluation of this component
    \param x   (i) x-coordinate of the origin of the coordinate system of this component
    \param y   (i) y-coordinate of the origin of the coordinate system of this component
    \param z   (i) z-coordinate of the origin of the coordinate system of this component

    */
    void AddExpr(std::string buf, double x, double y, double z);
    /*!

    \brief Return the number of components of this spatial function
    (1 for scalar functions, dim for vector-valued functions)

    \return number of components

    */
    virtual int NumberComponents()
      {
        return(expr_.size());
      };

  private:

    /*
      for scalar spatial functions returning the same value for all
      dimensions:

     -----------------------------------------+-------+-------+-------
      spatial function for dimension          |   0   |   1   |   2
     -----------------------------------------+-------+-------+-------
      origin of spatial function, x-component |         x_[0]
     -----------------------------------------+-----------------------
      origin of spatial function, y-component |         y_[0]
     -----------------------------------------+-------+-------+-------
      origin of spatial function, x-component |         z_[0]
     -----------------------------------------+-------+-------+-------


      for vector-valued spatial functions, returning seperate values
      for all dimensions:

     -----------------------------------------+-------+-------+-------
      spatial function for dimension          |   0   |   1   |   2
     -----------------------------------------+-------+-------+-------
      origin of spatial function, x-component | x_[0] | x_[1] | x_[2]
     -----------------------------------------+-------+-------+-------
      origin of spatial function, y-component | y_[0] | y_[1] | y_[2]
     -----------------------------------------+-------+-------+-------
      origin of spatial function, x-component | z_[0] | z_[1] | z_[2]
     -----------------------------------------+-------+-------+-------

    */


    std::vector<double> x_; //! origin(s) of spatial function, x-component
    std::vector<double> y_; //! origin(s) of spatial function, y-component
    std::vector<double> z_; //! origin(s) of spatial function, z-component

    std::vector<Teuchos::RCP<DRT::PARSER::Parser<double> > > expr_; //! expression syntax tree(s)
  };


  /// special implementation for 3d Beltrami flow
  class BeltramiFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w,p)

    */
    virtual int NumberComponents()
      {
        return(4);
      };

  };


  /// special implementation for 2d Kim-Moin flow
  class KimMoinFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (this is a vector-valued functions)

    \return number of components (u,v,p)

    */
    virtual int NumberComponents()
      {
        return(3);
      };

  };


  /// special implementation for 2d Bochev test case (velocity and pressure)
  class BochevUPFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,p)

    */
    virtual int NumberComponents()
      {
        return(3);
      };

  };


  /// special implementation for 2d Bochev test case (rhs function)
  class BochevRHSFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (f1,f2)

    */
    virtual int NumberComponents()
      {
        return(2);
      };

  };



  /// special implementation for beltrami flow (rhs)
  class BeltramiRHS : public Function
  {
  public:


    BeltramiRHS(int mat_id, bool is_stationary, bool is_stokes);

    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w)

    */
    virtual int NumberComponents()
      {
        return(3);
      };

  private:
    double kinviscosity_;
    bool is_stationary_;
    bool is_stokes_;

  };



  /// special implementation for 2d(3D) stationary kim-moin flow (rhs) for pure stokes equation
  class KimMoinRHS : public Function
  {
  public:


    KimMoinRHS(int mat_id, bool is_stationary, bool is_stokes);

    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w)

    */
    virtual int NumberComponents()
      {
        return(3);
      };

  private:
    double kinviscosity_;
    bool is_stationary_;
    bool is_stokes_;

  };


  /// special implementation for (randomly) disturbed 3d turbulent
  /// boundary-layer profile
  /// (currently fixed for low-Mach-number flow through a backward-facing step,
  ///  but may be easily manipulated to fit other profiles in other geometries)
  class TurbBouLayerFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w,p)

    */
    virtual int NumberComponents()
      {
        return(4);
      };

  };


  /// special implementation for (randomly) disturbed 3d turbulent
  /// boundary-layer profile
  /// (incompressible flow over backward-facing step,
  ///  corresponding to geometry of DNS by Le, Moin and Kim)
  class TurbBouLayerFunctionBFS : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w,p)

    */
    virtual int NumberComponents()
      {
        return(4);
      };

  };


  /// special implementation for (randomly) disturbed 3d turbulent boundary-layer profile
  /// (incompressible flow in the ORACLES test rig)
  class TurbBouLayerFunctionORACLES : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

  };


  /// special implementation for Womersley blood flow
  class WomersleyFunction : public Function
  {
  public:

    // ctor
    WomersleyFunction(bool locsys, int e, int mat, int curve, bool fsi);


    // evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
    // Bessel Functions of the first kind and order 0 or 1 for a complex argument
    std::complex<double> BesselJ01(std::complex<double> z, bool order);
    // perform a discrete fourier transformation on given array
    void DFT(std::vector<double> *data, std::vector< std::complex<double> > *resdata, const int N);

  private:

    bool                   isinit_;
    // switch for use of local coordinate systems
    bool                   locsys_;
    int                    locsysid_;
    // current edge node radius
    double 								 radius_;
    // a time curve
    DRT::UTILS::TimeCurve& tc_;
    // number of the material in the input file
    int                    mat_;
    // time curve number
    int                    curve_;
    double                 viscosity_;
    // FSI switch
    bool									 fsi_;
    // toggle coordinate transformation of edge node (once per time step)
    bool 									 dotrafo_;
    // store t_(n-1) for comparison with t_n
    double              	 tnminus1_;

    // further variables
    // number of harmonics that are used in the synthesis of the timecurve
    int										 noharm_;
    // time curve frequency
    double								 fbase_;
		 // time curve value of the previous time step (needed in current version to circumvent division by 0)
		double								 tcprevious_;
    // imaginary number i
    std::complex<double>   i_;
    // storage vector for velocity@1s for profile transition 0<t<1
    std::vector<double>	   vtemp_;
    // storage vector for Fourier Transform output
    std::vector< std::complex<double> > fouphyscurve_;

    // exist after init phase if locsys_==true
    // to do: move this stuff to separate class in locsys.H/.cpp
    // with functionality to transform spatial vectors from/to local system
    Condition*             locsyscond_;
    std::vector<double>    normal_;
    std::vector<double>    tangent1_;
    std::vector<double>    tangent2_;
    std::vector<double>    origin_;

    // exist after init phase if dirich_==true
		//for edge nodes (polar coordinates)
			// location of smallest modulus of phi, phi<0
		int                    iminminus_;
			// location of smallest modulus of phi, phi>0
		int                    iminplus_;
			// location of largest modulus of phi, phi<0
		int                    imaxminus_;
			// location of largest modulus of phi, phi>0
		int                    imaxplus_;
		 // vector with inflow surface ids
		std::vector<int>       surfnodeids_;
		 // vector with edge node ids
		std::vector<int>       nodeids_;
			// phase vector
		std::vector<double>    phi_;
			// distances between center of cross section and nodes
		std::vector<double>    noderadius_;
  };


  /// special implementation for stationary 2d Jeffery-Hamel flow
  class JefferyHamelFlowFunction : public Function
  {
  public:
    /*!

    \brief evaluate function at given position in space

    \param index (i) index defines the function-component which will
                     be evaluated
    \param x     (i) The point in space in which the function will be
                     evaluated

    */
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

    /*!

    \brief Return the number of components of this spatial function
    (This is a vector-valued function)

    \return number of components (u,v,w,p)

    */
    virtual int NumberComponents()
      {
        return(4);
      };

    /*!

    compute radial flow u as a function of the angle alpha

    \return radial velocity u

    \note this function is static such that we can get the radial velocity
          without creating the Function object

    */
    static double RadialVelocity(
        const double& theta ///< angle between 0 and PI/4 (range is checked in debug mode)
        );

  };


  /// special implementation for a level set test function
  class ZalesaksDiskFunction : public Function
  {
  public:

    /// ctor
    ZalesaksDiskFunction();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for a combustion test function
  class CircularFlame2Function : public Function
  {
  public:

    /// ctor
    CircularFlame2Function();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for a combustion test function
  class CircularFlame3Function : public Function
  {
  public:

    /// ctor
    CircularFlame3Function();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for a combustion test function
  class CircularFlame4Function : public Function
  {
  public:

    /// ctor
    CircularFlame4Function();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation two-phase flow test case
  class CollapsingWaterColumnFunction : public Function
  {
  public:

    /// ctor
    CollapsingWaterColumnFunction();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation two-phase flow test case
  class CollapsingWaterColumnFunctionCoarse : public Function
  {
  public:

    /// ctor
    CollapsingWaterColumnFunctionCoarse();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for the G-function in the ORACLES problem
  class ORACLESGFunction : public Function
  {
  public:

    /// ctor
    ORACLESGFunction();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for a level set test function
  class RotatingConeFunction : public Function
  {
  public:

    /// ctor
    RotatingConeFunction();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for a xfem test function
  class LevelSetCutTestFunction : public Function
  {
  public:

    /// ctor
    LevelSetCutTestFunction();

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);
  };

  /// special implementation for controlled rotations
  class ControlledRotationFunction : public Function
  {
  public:

    /// ctor
    ControlledRotationFunction(std::string fileName, std::string type, double origin_x, double origin_y, double origin_z);

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

  private:
    // Condition type: STRUCTURE=1, FLUID=2
    int type_;

    // Origin, about which the rotation shall be performed
    LINALG::Matrix<3,1> origin_;

    // Time of previous time step (at t-deltaT)
    double timeOld_;

    // Number of maneuver cells (variables)
    const int NUMMANEUVERCELLS_;

    // Number of maneuvers
    int numManeuvers_;

    // Double Vector containing maneuver information (t, omegaDot_x_B, omegaDot_y_B, omegaDot_z_B)
    std::vector<double> maneuvers_;

    // Previous angular acceleration (at t-deltaT)
    LINALG::Matrix<3,1> omegaDotOld_B_;

    // Current angular rate (at t)
    LINALG::Matrix<3,1> omega_B_;

    // Satellite's current attitude trafo matrix from B- to I-system (at t)
    LINALG::Matrix<3,3> satAtt_dcm_IB_;

    // Satellite's current attitude quaternion from B- to I-system (at t)
    LINALG::Matrix<4,1> satAtt_q_IB_;
  };

  /// special implementation for acceleration profiles
  class AccelerationProfileFunction : public Function
  {
  public:

    /// ctor
    AccelerationProfileFunction(std::string fileName);

    /// evaluate function at given position in space
    double Evaluate(int index, const double* x, double t, DRT::Discretization* dis);

  private:
    // Time of previous time step (at t-deltaT)
    double timeOld_;

    // Number of cells (variables) (time + 3-dim acc = 4)
    const int NUMACCELERATIONCELLS_;

    // Number of acceleration rows
    int numAccelerations_;

    // Double Vector containing acceleration information (t, acc_x_B, acc_y_B, acc_z_B)
    std::vector<double> accelerations_;

    // Current acceleration (at t)
    LINALG::Matrix<3,1> acc_B_;
  };

}
}


/*----------------------------------------------------------------------*/
//! Print function
/*----------------------------------------------------------------------*/
void PrintFunctionDatHeader()
{
  DRT::UTILS::FunctionManager functionmanager;
  Teuchos::RCP<DRT::INPUT::Lines> lines = functionmanager.ValidFunctionLines();

  lines->Print(std::cout);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<DRT::INPUT::Lines> DRT::UTILS::FunctionManager::ValidFunctionLines()
{
  DRT::INPUT::LineDefinition linelin;
  linelin
    .AddNamedInt("FUNCT")
    .AddNamedDoubleVector("LINE_LIN",8)
    ;

  DRT::INPUT::LineDefinition linequad;
  linequad
    .AddNamedInt("FUNCT")
    .AddNamedDoubleVector("LINE_QUAD",6)
    ;

  DRT::INPUT::LineDefinition radiuslin;
  radiuslin
    .AddNamedInt("FUNCT")
    .AddNamedDoubleVector("RADIUS_LIN",8)
    ;

  DRT::INPUT::LineDefinition radiusquad;
  radiusquad
    .AddNamedInt("FUNCT")
    .AddNamedDoubleVector("RADIUS_QUAD",6)
    ;

  DRT::INPUT::LineDefinition beltrami;
  beltrami
    .AddNamedInt("FUNCT")
    .AddTag("BELTRAMI")
    ;

  DRT::INPUT::LineDefinition kimmoin;
  kimmoin
    .AddNamedInt("FUNCT")
    .AddTag("KIM-MOIN")
    ;

  DRT::INPUT::LineDefinition bochevup;
  bochevup
    .AddNamedInt("FUNCT")
    .AddTag("BOCHEV-UP")
    ;

  DRT::INPUT::LineDefinition bochevrhs;
  bochevrhs
    .AddNamedInt("FUNCT")
    .AddTag("BOCHEV-RHS")
    ;

  DRT::INPUT::LineDefinition beltramiup;
  beltramiup
    .AddNamedInt("FUNCT")
    .AddTag("BELTRAMI-UP")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    ;

  DRT::INPUT::LineDefinition beltramigradu;
  beltramigradu
    .AddNamedInt("FUNCT")
    .AddTag("BELTRAMI-GRADU")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    ;

  DRT::INPUT::LineDefinition beltramirhs;
  beltramirhs
    .AddNamedInt("FUNCT")
    .AddTag("BELTRAMI-RHS")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    .AddNamedInt("ISSTOKES")
    ;

  DRT::INPUT::LineDefinition kimmoinup;
  kimmoinup
    .AddNamedInt("FUNCT")
    .AddTag("KIMMOIN-UP")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    ;

  DRT::INPUT::LineDefinition kimmoingradu;
  kimmoingradu
    .AddNamedInt("FUNCT")
    .AddTag("KIMMOIN-GRADU")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    ;

  DRT::INPUT::LineDefinition kimmoinrhs;
  kimmoinrhs
    .AddNamedInt("FUNCT")
    .AddTag("KIMMOIN-RHS")
    .AddNamedInt("MAT")
    .AddNamedInt("ISSTAT")
    .AddNamedInt("ISSTOKES")
    ;

  DRT::INPUT::LineDefinition turbboulayer;
  turbboulayer
    .AddNamedInt("FUNCT")
    .AddTag("TURBBOULAYER")
    ;

  DRT::INPUT::LineDefinition turbboulayerbfs;
  turbboulayerbfs
    .AddNamedInt("FUNCT")
    .AddTag("TURBBOULAYER-BFS")
    ;

  DRT::INPUT::LineDefinition turbboulayeroracles;
  turbboulayeroracles
    .AddNamedInt("FUNCT")
    .AddTag("TURBBOULAYERORACLES")
    ;

  DRT::INPUT::LineDefinition jefferyhamel;
  jefferyhamel
    .AddNamedInt("FUNCT")
    .AddTag("JEFFERY-HAMEL")
    ;

  DRT::INPUT::LineDefinition womersley;
  womersley
    .AddNamedInt("FUNCT")
    .AddTag("WOMERSLEY")
    .AddNamedInt("Local")
    .AddNamedInt("MAT")
    .AddNamedInt("CURVE")
    .AddNamedString("FSI")
    ;

  DRT::INPUT::LineDefinition localwomersley;
  localwomersley
    .AddNamedInt("FUNCT")
    .AddTag("WOMERSLEY")
    .AddNamedInt("Local")
    .AddNamedDouble("Radius")
    .AddNamedInt("MAT")
    .AddNamedInt("CURVE")
    ;

  DRT::INPUT::LineDefinition cylinder3d;
  cylinder3d
    .AddNamedInt("FUNCT")
    .AddNamedDouble("CYLINDER_3D")
    ;

  DRT::INPUT::LineDefinition zalesaksdisk;
  zalesaksdisk
    .AddNamedInt("FUNCT")
    .AddTag("ZALESAKSDISK")
    ;

  DRT::INPUT::LineDefinition circularflame2;
  circularflame2
    .AddNamedInt("FUNCT")
    .AddTag("CIRCULARFLAME2")
    ;

  DRT::INPUT::LineDefinition circularflame3;
  circularflame3
    .AddNamedInt("FUNCT")
    .AddTag("CIRCULARFLAME3")
    ;

  DRT::INPUT::LineDefinition circularflame4;
  circularflame4
    .AddNamedInt("FUNCT")
    .AddTag("CIRCULARFLAME4")
    ;

  DRT::INPUT::LineDefinition collapsingwatercolumn;
  collapsingwatercolumn
    .AddNamedInt("FUNCT")
    .AddTag("COLLAPSINGWATERCOLUMN")
    ;

  DRT::INPUT::LineDefinition collapsingwatercolumncoarse;
  collapsingwatercolumncoarse
    .AddNamedInt("FUNCT")
    .AddTag("COLLAPSINGWATERCOLUMNCOARSE")
    ;

  DRT::INPUT::LineDefinition oraclesgfunc;
  oraclesgfunc
    .AddNamedInt("FUNCT")
    .AddTag("ORACLESGFUNC")
    ;

  DRT::INPUT::LineDefinition rotatingcone;
  rotatingcone
    .AddNamedInt("FUNCT")
    .AddTag("ROTATINGCONE")
    ;

  DRT::INPUT::LineDefinition levelsetcuttest;
  levelsetcuttest
    .AddNamedInt("FUNCT")
    .AddTag("LEVELSETCUTTEST")
    ;

  DRT::INPUT::LineDefinition controlledrotation;
  controlledrotation
    .AddNamedInt("FUNCT")
    .AddTag("CONTROLLEDROTATION")
    .AddNamedString("FILE")
    .AddNamedString("TYPE")
    .AddNamedDoubleVector("ORIGIN",3)
    ;

  DRT::INPUT::LineDefinition accelerationprofile;
  accelerationprofile
    .AddNamedInt("FUNCT")
    .AddTag("ACCELERATIONPROFILE")
    .AddNamedString("FILE")
    ;

  DRT::INPUT::LineDefinition componentexpr;
  componentexpr
    .AddNamedInt("FUNCT")
    .AddNamedInt("COMPONENT")
    .AddNamedDoubleVector("EXPR",3)
    .AddNamedString("FUNCTION")
    ;

  DRT::INPUT::LineDefinition expr;
  expr
    .AddNamedInt("FUNCT")
    .AddNamedDoubleVector("EXPR",3)
    .AddNamedString("FUNCTION")
    ;

  Teuchos::RCP<DRT::INPUT::Lines> lines = Teuchos::rcp(new DRT::INPUT::Lines("FUNCT"));
  lines->Add(linelin);
  lines->Add(linequad);
  lines->Add(radiuslin);
  lines->Add(radiusquad);
  lines->Add(beltrami);
  lines->Add(kimmoin);
  lines->Add(bochevup);
  lines->Add(bochevrhs);
  lines->Add(beltramiup);
  lines->Add(beltramigradu);
  lines->Add(beltramirhs);
  lines->Add(kimmoinup);
  lines->Add(kimmoingradu);
  lines->Add(kimmoinrhs);
  lines->Add(turbboulayer);
  lines->Add(turbboulayerbfs);
  lines->Add(turbboulayeroracles);
  lines->Add(jefferyhamel);
  lines->Add(womersley);
  lines->Add(localwomersley);
  lines->Add(cylinder3d);
  lines->Add(zalesaksdisk);
  lines->Add(circularflame2);
  lines->Add(circularflame3);
  lines->Add(circularflame4);
  lines->Add(collapsingwatercolumn);
  lines->Add(collapsingwatercolumncoarse);
  lines->Add(oraclesgfunc);
  lines->Add(rotatingcone);
  lines->Add(levelsetcuttest);
  lines->Add(controlledrotation);
  lines->Add(accelerationprofile);
  lines->Add(componentexpr);
  lines->Add(expr);
  return lines;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::FunctionManager::ReadInput(DRT::INPUT::DatFileReader& reader)
{
  functions_.clear();

  Teuchos::RCP<DRT::INPUT::Lines> lines = ValidFunctionLines();

  // test for as many functions as there are
  for (int i = 1;; ++i)
  {
    std::vector<Teuchos::RCP<DRT::INPUT::LineDefinition> > functions = lines->Read(reader, i);
    if (functions.size()==0)
      break;

    if (functions.size()==1)
    {
      Teuchos::RCP<DRT::INPUT::LineDefinition> function = functions[0];

      int id;
      function->ExtractInt("FUNCT",id);
      if (id!=i) dserror("expected FUNCT%d but got FUNCT%d", i, id);

      if (function->HaveNamed("LINE_LIN"))
      {
        std::vector<double> tmp;
        function->ExtractDoubleVector("LINE_LIN",tmp);

        double x1[3];
        double x2[3];
        x1[0] = tmp[0];
        x1[1] = tmp[1];
        x1[2] = tmp[2];
        //double val1 = tmp[3];
        x2[0] = tmp[4];
        x2[1] = tmp[5];
        x2[2] = tmp[6];
        //double val2 = tmp[7];

        /* calculate slope and offset */
        double b = tmp[3];
        double m = (tmp[7]-tmp[3]);
        double length = sqrt((tmp[4]-tmp[0])*(tmp[4]-tmp[0]) +
                             (tmp[5]-tmp[1])*(tmp[5]-tmp[1]) +
                             (tmp[6]-tmp[2])*(tmp[6]-tmp[2]));

        // Keep it simple. We use the expression based function class
        // that is able to handle straight lines.
        std::ostringstream expr;
        expr << "(" << b << ") + ((" << x2[0]-x1[0] << ")*x + (" << x2[1]-x1[1] << ")*y + (" << x2[2]-x1[2] << ")*z)/(" << length << ")/(" << length << ")*(" << m << ")";

        functions_.push_back(Teuchos::rcp(new ExprFunction(const_cast<char*>(expr.str().c_str()), x1[0], x1[1], x1[2])));
      }
      else if (function->HaveNamed("LINE_QUAD"))
      {
        std::vector<double> tmp;
        function->ExtractDoubleVector("LINE_QUAD",tmp);

        double x1[3];
        double x2[3];
        x1[0] = tmp[0];
        x1[1] = tmp[1];
        x1[2] = tmp[2];
        x2[0] = tmp[3];
        x2[1] = tmp[4];
        x2[2] = tmp[5];

        /* calculate length */
        double length = sqrt((tmp[3]-tmp[0])*(tmp[3]-tmp[0]) +
                             (tmp[4]-tmp[1])*(tmp[4]-tmp[1]) +
                             (tmp[5]-tmp[2])*(tmp[5]-tmp[2]));

        // Keep it simple.
        std::ostringstream expr;
        expr << "1.0 - 4 * (((" << x2[0]-x1[0] << ")*x + (" << x2[1]-x1[1] << ")*y + (" << x2[2]-x1[2] << ")*z)/(" << length << ")/(" << length << ") - 1.0/2.0)^2.0";
        functions_.push_back(Teuchos::rcp(new ExprFunction(const_cast<char*>(expr.str().c_str()), x1[0], x1[1], x1[2])));
      }
      else if (function->HaveNamed("RADIUS_LIN"))
      {
        std::vector<double> tmp;
        function->ExtractDoubleVector("RADIUS_LIN",tmp);

        double x1[3];
        x1[0] = tmp[0];
        x1[1] = tmp[1];
        x1[2] = tmp[2];

        /* calculate slope and offset */
        double b = tmp[3];
        double m = (tmp[7]-tmp[3]);
        double length = sqrt((tmp[4]-tmp[0])*(tmp[4]-tmp[0]) +
                             (tmp[5]-tmp[1])*(tmp[5]-tmp[1]) +
                             (tmp[6]-tmp[2])*(tmp[6]-tmp[2]));

        // Keep it simple.
        std::ostringstream expr;
        expr << "(" << b << ") + sqrt(x*x + y*y + z*z)/(" << length << ")*(" << m << ")";
        functions_.push_back(Teuchos::rcp(new ExprFunction(const_cast<char*>(expr.str().c_str()), x1[0], x1[1], x1[2])));
      }
      else if (function->HaveNamed("RADIUS_QUAD"))
      {
        std::vector<double> tmp;
        function->ExtractDoubleVector("RADIUS_QUAD",tmp);

        double x1[3];

        x1[0] = tmp[0];
        x1[1] = tmp[1];
        x1[2] = tmp[2];

        /* calculate length */
        double length = sqrt((tmp[3]-tmp[0])*(tmp[3]-tmp[0]) +
                             (tmp[4]-tmp[1])*(tmp[4]-tmp[1]) +
                             (tmp[5]-tmp[2])*(tmp[5]-tmp[2]));

        // Keep it simple.
        std::ostringstream expr;
        expr << "1.0 - (x*x + y*y + z*z)/(" << length << ")/(" << length << ")";
        functions_.push_back(Teuchos::rcp(new ExprFunction(const_cast<char*>(expr.str().c_str()), x1[0], x1[1], x1[2])));
      }
      else if (function->HaveNamed("BELTRAMI"))
      {
        functions_.push_back(Teuchos::rcp(new BeltramiFunction()));
      }
      else if (function->HaveNamed("KIM-MOIN"))
      {
        functions_.push_back(Teuchos::rcp(new KimMoinFunction()));
      }
      else if (function->HaveNamed("BOCHEV-UP"))
      {
        functions_.push_back(Teuchos::rcp(new BochevUPFunction()));
      }
      else if (function->HaveNamed("BOCHEV-RHS"))
      {
        functions_.push_back(Teuchos::rcp(new BochevRHSFunction()));
      }
      else if (function->HaveNamed("BELTRAMI-UP"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in BELTRAMI-UP");

        functions_.push_back(Teuchos::rcp(new BeltramiUP(mat_id, (bool)is_stationary)));
      }
      else if (function->HaveNamed("BELTRAMI-GRADU"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in BELTRAMI-GRADU");

        functions_.push_back(Teuchos::rcp(new BeltramiGradU(mat_id, (bool)is_stationary)));
      }
      else if (function->HaveNamed("BELTRAMI-RHS"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;
        int is_stokes     = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);
        function->ExtractInt("ISSTOKES",is_stokes);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in BELTRAMI-GRADU");

        functions_.push_back(Teuchos::rcp(new BeltramiRHS(mat_id, (bool)is_stationary, (bool)is_stokes)));
      }
      else if (function->HaveNamed("KIMMOIN-UP"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in KIMMOIN-UP");

        functions_.push_back(Teuchos::rcp(new KimMoinUP(mat_id, (bool)is_stationary)));
      }
      else if (function->HaveNamed("KIMMOIN-GRADU"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in KIMMOIN-GRADU");

        functions_.push_back(Teuchos::rcp(new KimMoinGradU(mat_id, (bool)is_stationary)));
      }
      else if (function->HaveNamed("KIMMOIN-RHS"))
      {
        // read material
        int mat_id = -1;
        int is_stationary = 0;
        int is_stokes     = 0;

        function->ExtractInt("MAT",mat_id);
        function->ExtractInt("ISSTAT",is_stationary);
        function->ExtractInt("ISSTOKES",is_stokes);

        if(mat_id<=0) dserror("Please give a (reasonable) 'MAT'/material in KIMMOIN-GRADU");

        functions_.push_back(Teuchos::rcp(new KimMoinRHS(mat_id, (bool)is_stationary, (bool)is_stokes)));
      }
      else if (function->HaveNamed("TURBBOULAYER"))
      {
        functions_.push_back(Teuchos::rcp(new TurbBouLayerFunction()));
      }
      else if (function->HaveNamed("TURBBOULAYER-BFS"))
      {
        functions_.push_back(Teuchos::rcp(new TurbBouLayerFunctionBFS()));
      }
      else if (function->HaveNamed("TURBBOULAYERORACLES"))
      {
        functions_.push_back(Teuchos::rcp(new TurbBouLayerFunctionORACLES()));
      }
      else if (function->HaveNamed("JEFFERY-HAMEL"))
      {
        functions_.push_back(Teuchos::rcp(new JefferyHamelFlowFunction()));
      }
      else if (function->HaveNamed("WOMERSLEY"))
      {
        int e = -1;
        bool localcoordsystem = false;
        if (function->HaveNamed("Local"))
        {
          localcoordsystem = true;
          function->ExtractInt("Local",e);
        }
        else
        	dserror("Please give a number of an existing local coordinate system in the Womersley function definition under 'Local'!");
        // read material
        int mat = -1;
        function->ExtractInt("MAT",mat);
				if(mat<=0) dserror("Please give a (reasonable) 'MAT'/material in WOMERSLEY FUNCT");
				// read curve
        int curve = -1;
        function->ExtractInt("CURVE",curve);
        if(curve<=0) dserror("Please give a (resonable) 'CURVE' in WOMERSLEY FUNCT");
				// read option FSI
        bool fsi = false;
        if(function->HaveNamed("FSI"))
        {
        	std::string read;
        	function->ExtractString("FSI", read);
        	if(read=="Yes")
        		fsi = true;
        	else
        		fsi = false;
        }

        functions_.push_back(Teuchos::rcp(new WomersleyFunction(localcoordsystem,e-1,mat,curve-1,fsi)));
      }
      else if (function->HaveNamed("CYLINDER_3D"))
      {
        double um;
        function->ExtractDouble("CYLINDER_3D", um);
        double h = 0.41;

        // Keep it simple.
        // This is particularly odd. Very special. Useless.
        std::ostringstream expr;
        expr << "16*(" << um << ")*y*z*((" << h << ")-y)*((" << h << ")-z) / ((" << h << ")^4)";
        functions_.push_back(Teuchos::rcp(new ExprFunction(const_cast<char*>(expr.str().c_str()), 0, 0, 0)));
      }
      else if (function->HaveNamed("ZALESAKSDISK"))
      {
        functions_.push_back(Teuchos::rcp(new ZalesaksDiskFunction()));
      }
      else if (function->HaveNamed("CIRCULARFLAME2"))
      {
        functions_.push_back(Teuchos::rcp(new CircularFlame2Function()));
      }
      else if (function->HaveNamed("CIRCULARFLAME3"))
      {
        functions_.push_back(Teuchos::rcp(new CircularFlame3Function()));
      }
      else if (function->HaveNamed("CIRCULARFLAME4"))
      {
        functions_.push_back(Teuchos::rcp(new CircularFlame4Function()));
      }
      else if (function->HaveNamed("COLLAPSINGWATERCOLUMN"))
      {
        functions_.push_back(Teuchos::rcp(new CollapsingWaterColumnFunction()));
      }
      else if (function->HaveNamed("COLLAPSINGWATERCOLUMNCOARSE"))
      {
        functions_.push_back(Teuchos::rcp(new CollapsingWaterColumnFunctionCoarse()));
      }
      else if (function->HaveNamed("ORACLESGFUNC"))
      {
        functions_.push_back(Teuchos::rcp(new ORACLESGFunction()));
      }
      else if (function->HaveNamed("ROTATINGCONE"))
      {
        functions_.push_back(Teuchos::rcp(new RotatingConeFunction()));
      }
      else if (function->HaveNamed("LEVELSETCUTTEST"))
      {
        functions_.push_back(Teuchos::rcp(new LevelSetCutTestFunction()));
      }
      else if (function->HaveNamed("CONTROLLEDROTATION"))
      {
          std::string fileName;
          function->ExtractString("FILE", fileName);

          std::string type;
          function->ExtractString("TYPE", type);

          std::vector<double> origin;
          function->ExtractDoubleVector("ORIGIN",origin);

          functions_.push_back(Teuchos::rcp(new ControlledRotationFunction(fileName, type, origin[0], origin[1], origin[2])));
      }
      else if (function->HaveNamed("ACCELERATIONPROFILE"))
      {
          std::string fileName;
          function->ExtractString("FILE", fileName);

          functions_.push_back(Teuchos::rcp(new AccelerationProfileFunction(fileName)));
      }
      else if (function->HaveNamed("EXPR"))
      {
        Teuchos::RCP<ExprFunction> vecfunc = Teuchos::rcp(new ExprFunction());

        std::vector<double> origin;
        function->ExtractDoubleVector("EXPR",origin);
        std::string component;
        function->ExtractString("FUNCTION",component);

        vecfunc->AddExpr(component,origin[0],origin[1],origin[2]);
        functions_.push_back(vecfunc);
      }
      else
        dserror("unrecognized function");
    }

    else
    {
      Teuchos::RCP<ExprFunction> vecfunc = Teuchos::rcp(new ExprFunction());

      for (unsigned j=0; j<functions.size(); ++j)
      {
        int id;
        functions[j]->ExtractInt("FUNCT",id);
        if (id!=i) dserror("expected FUNCT%d but got FUNCT%d", i, id);

        if (not functions[j]->HaveNamed("COMPONENT"))
          dserror("component based expression function expected");

        int dim;
        functions[j]->ExtractInt("COMPONENT", dim);

        if (dim!=static_cast<int>(j))
        {
          dserror("For vector valued functions the components have to be\n"
                  "specified succesively, e.g. 0,1,..,ndof");
        }

        std::vector<double> origin;
        functions[j]->ExtractDoubleVector("EXPR",origin);
        std::string component;
        functions[j]->ExtractString("FUNCTION",component);

        vecfunc->AddExpr(component,origin[0],origin[1],origin[2]);
      }
      functions_.push_back(vecfunc);
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::Function& DRT::UTILS::FunctionManager::Funct(int num)
{
  // ensure that desired function is available (prevents segmentation fault)
  if (functions_.size()< (unsigned int)(num+1) || num<0)
    dserror("function %d not available",num+1);

  return *(functions_[num]);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::ExprFunction::ExprFunction()
{
  x_.clear();
  y_.clear();
  z_.clear();

  expr_.clear();

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::ExprFunction::ExprFunction(char* buf,
                                       double x,
                                       double y,
                                       double z)
{

  x_.push_back(x);
  y_.push_back(y);
  z_.push_back(z);

  expr_.push_back(Teuchos::rcp(new DRT::PARSER::Parser<double>(buf)));

  return;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
DRT::UTILS::ExprFunction::~ExprFunction()
{
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void DRT::UTILS::ExprFunction::AddExpr(std::string buf,
                                       double x,
                                       double y,
                                       double z
  )
{
  expr_.push_back(Teuchos::rcp(new DRT::PARSER::Parser<double>(buf)));
  x_.push_back(x);
  y_.push_back(y);
  z_.push_back(z);

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::ExprFunction::Evaluate(int index, const double* x, double t, DRT::Discretization* dis)
{
  // single expression for all components. Reset index to 0!
  if(expr_.size()==1)
  {
    index=0;
  }

  if(index>(int)expr_.size()-1 || index<0)
  {
    dserror("Tried to evaluate a function in a not available dimension.\nSpecify either one function or functions for all dimensions! \n(including one for the pressure)");
  }
  return expr_[index]->EvaluateFunct(x[0]-x_[index], x[1]-y_[index], x[2]-z_[index],t);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BeltramiFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double a = M_PI/4.0;
  double d = M_PI/2.0;

  int id = DRT::Problem::Instance()->Materials()->FirstIdByType(INPAR::MAT::m_fluid);
  if (id==-1) dserror("Newtonian fluid material could not be found");
  const MAT::PAR::Parameter* mat = DRT::Problem::Instance()->Materials()->ParameterById(id);
  const MAT::PAR::NewtonianFluid* actmat = static_cast<const MAT::PAR::NewtonianFluid*>(mat);
  double dens = actmat->density_;

  switch (index)
  {
  case 0:
    return -a * ( exp(a*xp[0]) * sin(a*xp[1] + d*xp[2]) +
                  exp(a*xp[2]) * cos(a*xp[0] + d*xp[1]) );
  case 1:
    return -a * ( exp(a*xp[1]) * sin(a*xp[2] + d*xp[0]) +
                  exp(a*xp[0]) * cos(a*xp[1] + d*xp[2]) );
  case 2:
    return -a * ( exp(a*xp[2]) * sin(a*xp[0] + d*xp[1]) +
                  exp(a*xp[1]) * cos(a*xp[2] + d*xp[0]) );
  case 3:
    return -a*a/2 * dens * ( exp(2*a*xp[0]) + exp(2*a*xp[1]) + exp(2*a*xp[2])
                      + 2* sin(a*xp[0]+d*xp[1]) * cos(a*xp[2]+d*xp[0]) * exp(a*(xp[1]+xp[2]))
                      + 2* sin(a*xp[1]+d*xp[2]) * cos(a*xp[0]+d*xp[1]) * exp(a*(xp[2]+xp[0]))
                      + 2* sin(a*xp[2]+d*xp[0]) * cos(a*xp[1]+d*xp[2]) * exp(a*(xp[0]+xp[1])));
  default:
    return 1.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::KimMoinFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double a = 2.0;

  switch (index)
  {
  case 0:
    return - cos(a*PI*xp[0]) * sin(a*PI*xp[1]);
  case 1:
    return + sin(a*PI*xp[0]) * cos(a*PI*xp[1]);
  case 2:
    return -1.0/4.0 * ( cos(2.0*a*PI*xp[0]) + cos(2.0*a*PI*xp[1]) );
  default:
    return 1.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BochevUPFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  switch (index)
  {
  case 0:
    return sin(PI*xp[0]-0.7)*sin(PI*xp[1]+0.2);
  case 1:
    return cos(PI*xp[0]-0.7)*cos(PI*xp[1]+0.2);
  case 2:
    return sin(xp[0])*cos(xp[1])+(cos(1.0)-1.0)*sin(1.0);
  default:
    return 1.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BochevRHSFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  switch (index)
  {
  case 0:
    return cos(xp[0])*cos(xp[1])+PI*sin(PI*xp[0]-0.7)*cos(PI*xp[0]-0.7)+2*PI*PI*sin(PI*xp[0]-0.7)*sin(PI*xp[1]+0.2);
  case 1:
    return -sin(xp[0])*sin(xp[1])-PI*sin(PI*xp[1]+0.2)*cos(PI*xp[1]+0.2)+2*PI*PI*cos(PI*xp[0]-0.7)*cos(PI*xp[1]+0.2);
  default:
    return 1.0;
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::BeltramiUP::BeltramiUP(int mat_id, bool is_stationary) :
Function(),
density_(-999.0e99),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get density
  density_ = fparams->density_;

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / density_;

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::BeltramiUP::BeltramiUP( Teuchos::RCP<MAT::Material> & mat, bool is_stationary) :
Function(),
density_(-999.0e99),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  if (mat->MaterialType() != INPAR::MAT::m_fluid)
    dserror("Material is not a fluid");
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get density
  density_ = fparams->density_;

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BeltramiUP::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double x = xp[0];
  double y = xp[1];
  double z = xp[2];

  double a = PI/4.0;
  double b = PI/4.0;
  double c= a*a + b*b + a*b;

  double K1 = exp(a*(x-z) + b*(y-z)); // =K4
  double K2 = exp(a*(z-y) + b*(x-y)); // =K5
  double K3 = exp(a*(y-x) + b*(z-x)); // =K6


  switch (index)
  {
  case 0:
    return b*K1-a*K2;
  case 1:
    return b*K3-a*K1;
  case 2:
    return b*K2-a*K3;
  case 3:
    return  c*( 1.0/K3 + 1.0/K2 + 1.0/K1 )* density_;
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::BeltramiGradU::BeltramiGradU(int mat_id, bool is_stationary ) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::BeltramiGradU::BeltramiGradU(Teuchos::RCP<MAT::Material> & mat, bool is_stationary ) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  if (mat->MaterialType() != INPAR::MAT::m_fluid)
    dserror("Material is not a fluid");
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BeltramiGradU::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double x = xp[0];
  double y = xp[1];
  double z = xp[2];

  double a = PI/4.0;
  double b = PI/4.0;

  double K1 = exp(a*(x-z) + b*(y-z)); // =K4
  double K2 = exp(a*(z-y) + b*(x-y)); // =K5
  double K3 = exp(a*(y-x) + b*(z-x)); // =K6


  switch (index)
  {
  case 0: // u,x
    return a*b*(K1-K2);
  case 1: // u,y
    return b*b*K1+a*(a+b)*K2;
  case 2: // u,z
    return -b*(a+b)*K1-a*a*K2;
  case 3: // v,x
    return  -b*(a+b)*K3-a*a*K1;
  case 4: // v,y
    return a*b*K3-a*b*K1;
  case 5: // v,z
    return b*b*K3+a*(a+b)*K1;
  case 6: // w,x
    return b*b*K2+a*(a+b)*K3;
  case 7: // w,y
    return -b*(a+b)*K2-a*a*K3;
  case 8: // w,z
    return a*b*(K2-K3);
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::BeltramiRHS::BeltramiRHS(int mat_id, bool is_stationary, bool is_stokes) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary),
is_stokes_(is_stokes)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::BeltramiRHS::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double x = xp[0];
  double y = xp[1];
  double z = xp[2];

  double a = PI/4.0;
  double b = PI/4.0;
  double c= a*a + b*b + a*b;

  double K1 = exp(a*(x-z) + b*(y-z)); // =K4
  double K2 = exp(a*(z-y) + b*(x-y)); // =K5
  double K3 = exp(a*(y-x) + b*(z-x)); // =K6

  double t1 = b*K1-a*K2;
  double t2 = b*K3-a*K1;
  double t3 = b*K2-a*K3;

  double conv_x = 0.0;
  double conv_y = 0.0;
  double conv_z = 0.0;

  if(!is_stokes_)
  {
    conv_x = t1 * (     a*b*K1 -     a*b*K2) + t2 * (     b*b*K1 + a*(a+b)*K2) + t3 * (-b*(a+b)*K1 -     a*a*K2);
    conv_y = t1 * (-b*(a+b)*K3 -     a*a*K1) + t2 * (     a*b*K3 -     a*b*K1) + t3 * (     b*b*K3 + a*(a+b)*K1);
    conv_z = t1 * (     b*b*K2 + a*(a+b)*K3) + t2 * (-b*(a+b)*K2 -     a*a*K3) + t3 * (     a*b*K2 -     a*b*K3);
  }

  switch (index)
  {
  case 0:
    return c*( (a+b)/K3 - b/K2 - a/K1 - 2.*kinviscosity_*(b*K1-a*K2) ) + conv_x;
  case 1:
    return c*( -a/K3 + (a+b)/K2 - b/K1 - 2.*kinviscosity_*(b*K3-a*K1) ) + conv_y;
  case 2:
    return c*( -b/K3 - a/K2 + (a+b)/K1 - 2.*kinviscosity_*(b*K2-a*K3) ) + conv_z;
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;

}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::KimMoinUP::KimMoinUP(int mat_id, bool is_stationary) :
Function(),
density_(-999.0e99),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get density
  density_ = fparams->density_;

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / density_;

}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::KimMoinUP::KimMoinUP( Teuchos::RCP<MAT::Material> & mat, bool is_stationary) :
Function(),
density_(-999.0e99),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  if (mat->MaterialType() != INPAR::MAT::m_fluid)
    dserror("Material is not a fluid");
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get density
  density_ = fparams->density_;

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::KimMoinUP::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  double x = xp[0];
  double y = xp[1];
  //double z = xp[2];

  double a = 2.0;

  double a_pi_x = a*PI*x;
  double a_pi_y = a*PI*y;

  // time factors
  double gu = 1.0;
  double gp = 1.0;

  if(!is_stationary_)
  {
    gu = exp(-2.0*a*a*PI*PI*t*kinviscosity_);
    gp = exp(-4.0*a*a*PI*PI*t*kinviscosity_);
  }

  switch (index)
  {
  case 0:
    return -cos(a_pi_x)*sin(a_pi_y) * gu;
  case 1:
    return sin(a_pi_x)*cos(a_pi_y) * gu;
  case 2:
    return 0.0;
  case 3:
    return  -1./4. * ( cos(2.0*a_pi_x) + cos(2.0*a_pi_y) ) * gp * density_;
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::KimMoinGradU::KimMoinGradU(int mat_id, bool is_stationary ) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::KimMoinGradU::KimMoinGradU(Teuchos::RCP<MAT::Material> & mat, bool is_stationary ) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary)
{

  if (mat->MaterialType() != INPAR::MAT::m_fluid)
    dserror("Material is not a fluid");
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::KimMoinGradU::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //double visc = 1.0;

  double x = xp[0];
  double y = xp[1];
  //double z = xp[2];

  double a = 2.0;

  double a_pi_x = a*PI*x;
  double a_pi_y = a*PI*y;

  // time factors
  double gu = 1.0;

  if(!is_stationary_)
  {
    gu = exp(-2.0*a*a*PI*PI*t*kinviscosity_);
  }


  switch (index)
  {
  case 0: // u,x
    return  sin(a_pi_x)*sin(a_pi_y)*a*PI * gu;
  case 1: // u,y
    return -cos(a_pi_x)*cos(a_pi_y)*a*PI * gu;
  case 2: // u,z
    return 0.0;
  case 3: // v,x
    return  cos(a_pi_x)*cos(a_pi_y)*a*PI * gu;
  case 4: // v,y
    return -sin(a_pi_x)*sin(a_pi_y)*a*PI * gu;
  case 5: // v,z
    return 0.0;
  case 6: // w,x
    return 0.0;
  case 7: // w,y
    return 0.0;
  case 8: // w,z
    return 0.0;
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
DRT::UTILS::KimMoinRHS::KimMoinRHS(int mat_id, bool is_stationary, bool is_stokes) :
Function(),
kinviscosity_(-999.0e99),
is_stationary_(is_stationary),
is_stokes_(is_stokes)
{

  // get material parameters for fluid
  Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_id);
  if (mat->Type() != INPAR::MAT::m_fluid)
    dserror("Material %d is not a fluid",mat_id);
  MAT::PAR::Parameter* params = mat->Parameter();
  MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
  if (!fparams)
    dserror("Material does not cast to Newtonian fluid");

  // get kinematic viscosity
  kinviscosity_ = fparams->viscosity_ / fparams->density_;

}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::KimMoinRHS::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double x = xp[0];
  double y = xp[1];
  //double z = xp[2];

  double a = 2.0;

  // time factors
  double gu = 1.0;
  double gp = 1.0;

  if(!is_stationary_)
  {
    gu = exp(-2.0*a*a*PI*PI*t*kinviscosity_);
    gp = exp(-4.0*a*a*PI*PI*t*kinviscosity_);
  }


  double a_pi_x = a*PI*x;
  double a_pi_y = a*PI*y;

  double conv_x = 0.0;
  double conv_y = 0.0;
  //double conv_z = 0.0;

  if(!is_stokes_)
  {
    conv_x = -a*PI*sin(a_pi_x)*cos(a_pi_x);
    conv_y = -a*PI*sin(a_pi_y)*cos(a_pi_y);
  }

  double visc_x = 0.0;
  double visc_y = 0.0;

  if(is_stationary_)
  {
    visc_x = kinviscosity_*2.*a*a*PI*PI* (-cos(a_pi_x)*sin(a_pi_y)); // * (gu = 1) for stationary
    visc_y = kinviscosity_*2.*a*a*PI*PI* ( sin(a_pi_x)*cos(a_pi_y)); // * (gu = 1) for stationary
  }

  // in case of instationary: du/dt - \nu \laplacian(u) = 0

  switch (index)
  {
  case 0:
    return 0.5*a*PI*sin(2.*a_pi_x)*gp + visc_x + conv_x*gu*gu;
  case 1:
    return 0.5*a*PI*sin(2.*a_pi_y)*gp + visc_y + conv_y*gu*gu;
  case 2:
    return 0.0;
  default:
    dserror("wrong index %d", index);
    break;
  }

  return 1.0;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::TurbBouLayerFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  switch (index)
  {
    double randomnumber;
    double noise;

    case 0:
    {
      // set u_tau und nu (fixed for low-Mach-number flow through
      // backward-facing step, for the time being) and compute l_tau
      double utau = 0.106;
      double nu   = 0.00001527;
      double ltau = nu/utau;

      // compute y+ (upper wall of backward-facing step fixed, for the time being)
      double yplus = xp[1]/ltau;
      double myplus = (0.082-xp[1])/ltau;

      // generate noise via random number
      randomnumber = DRT::Problem::Instance()->Random()->Uni();
      noise = 0.1778 * randomnumber;

      // return velocity value in x-direction
      if      (yplus <= 5.0)
        return utau * yplus + noise;
      else if ((yplus > 5.0)   and (yplus <= 30.0))
        return utau * ( 5.0 * log(yplus) - 3.05 ) + noise;
      else if ((yplus > 30.0)  and (yplus <= 285.0))
      {
        double upre = utau * ( 2.5 * log(yplus) + 5.5 );
        if (upre > 2.063) return 2.063 + noise;
        else             return upre + noise;
      }
      else if ((myplus > 30.0) and (myplus <= 284.0))
      {
        double upre = utau * ( 2.5 * log(myplus) + 5.5 );
        if (upre > 2.063) return 2.063 + noise;
        else              return upre + noise;
      }
      else if ((myplus > 5.0)  and (myplus <= 30.0))
        return utau * ( 5.0 * log(myplus) - 3.05 ) + noise;
      else if (myplus <= 5.0)
        return utau * myplus + noise;
    }
    case 1:
    {
      // generate noise via random number
      randomnumber = DRT::Problem::Instance()->Random()->Uni();
      noise = 0.1778 * randomnumber;

      // return velocity value in y-direction
      return noise;
    }
    case 2:
    {
      // generate noise via random number
      randomnumber = DRT::Problem::Instance()->Random()->Uni();
      noise = 0.1778 * randomnumber;

      // return velocity value in z-direction
      return noise;
    }
    // nothing to be done for hydrodynamic pressure values
    case 3:
  default:
    return 0.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::TurbBouLayerFunctionBFS::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double randomnumber = 0.0;
  double noise = 0.0;
  // maximal velocity
  const double max = 1.899;
  // fluctuation 10% of maximal velocity
  const double fluc = 0.1 * max;
  // generate noise in turbulent boundary layer only
  const bool boulayer = true;
  // step height
  double h = 0.041;
  // boundary layer thickness
  double delta = 1.2 * h;

  switch (index)
  {
    case 0:
    {
      // set u_tau und nu and compute l_tau
      double utau = 0.093;
      double nu   = 0.000015268;
      double ltau = nu/utau;

      // compute y+
      double yplus = xp[1]/ltau;

      if (!boulayer)
      {
        // noise over complete inlet
        // generate noise via random number between -1 and 1
        randomnumber = DRT::Problem::Instance()->Random()->Uni();
        noise = fluc * randomnumber;
      }
      else
      {
        // noise only in boundary layer
        if (xp[1]<=delta)
        {
          // generate noise via random number between -1 and 1
          randomnumber = DRT::Problem::Instance()->Random()->Uni();
          noise = fluc * randomnumber;
        }
        else
        {
          noise = 0.0;
        }
      }

      // return velocity value in x-direction
      // viscous sublayer
      if (yplus <= 5.0)
        return utau * yplus + noise;
      // buffer layer
      else if ((yplus > 5.0) and (yplus <= 30.0))
        return utau * ( 4.5 * log(yplus) - 2.2 ) + noise;
      // log-layer
      else if (yplus > 30.0 and (yplus <= 150.0))
        return utau * ( 2.6 * log(yplus) + 4.3 ) + noise;
      // defect law
      else if (yplus > 150.0)
      {
        double upre = utau * ( 3.6 * log(xp[1]/delta) - 0.35 ) + max;
        if (upre > max) return max + noise;
        else            return upre + noise;
      }
    }
    case 1:
    {
      if (!boulayer)
      {
        // noise over complete inlet
        // generate noise via random number between -1 and 1
        randomnumber = DRT::Problem::Instance()->Random()->Uni();
        noise = fluc * randomnumber;
      }
      else
      {
        // noise only in boundary layer
        if (xp[1]<=delta)
        {
          // generate noise via random number between -1 and 1
          randomnumber = DRT::Problem::Instance()->Random()->Uni();
          noise = fluc * randomnumber;
        }
        else
        {
          noise = 0.0;
        }
      }

      // return velocity value in y-direction
      return noise;
    }
    case 2:
    {
      if (!boulayer)
      {
        // noise over complete inlet
        // generate noise via random number between -1 and 1
        randomnumber = DRT::Problem::Instance()->Random()->Uni();
        noise = fluc * randomnumber;
      }
      else
      {
        // noise only in boundary layer
        if (xp[1]<=delta)
        {
          // generate noise via random number between -1 and 1
         randomnumber = DRT::Problem::Instance()->Random()->Uni();
          noise = fluc * randomnumber;
        }
        else
        {
          noise = 0.0;
        }
      }

      // return velocity value in z-direction
      return noise;
    }
    // nothing to be done for hydrodynamic pressure values
    case 3:
  default:
    return 0.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::TurbBouLayerFunctionORACLES::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double randomnumber;
  double noise;

  // fluctuation 10% of bulk mean velocity
  double fluc= 0.1;
  // maximal velocity
  double umax = 15.62;

  switch (index)
  {
    case 0:
    {
      // set u_tau und nu and compute l_tau
      const double utau = 0.7;
      const double nu   = 1.375E-5;
      double ltau = nu/utau;

      // transform global y-coordinate to local (upper and lower) channel y-coordinates
      double y=0.0;
      // remark: no tolerances needed, since there will always be no-slip boundary conditions
      // upper half upper inlet channel
      if (xp[1]>0.0202 and xp[1]<0.0354+1.0E-9)
        y = -(xp[1]-0.0354);
      // lower half upper inlet channel
      else if (xp[1]>0.005-1.0E-9 and xp[1]<=0.0202)
        y = xp[1]-0.005;
      // upper half lower inlet channel
      else if (xp[1]>=-0.0202 and xp[1]<-0.005+1.0E-9)
        y = -(xp[1]+0.005);
      // lower half lower inlet channel
      else if (xp[1]>-0.0354-1.0E-9 and xp[1]<-0.0202)
        y = xp[1]+0.0354;
      else
      {
        std::cout << xp[1] << std::endl;
        //dserror("coordinates do not match to ORACLES problem");
      }
      // compute y+
      double yplus = y/ltau;

      // generate noise via random number between -1 and +1
      randomnumber = DRT::Problem::Instance()->Random()->Uni();
      noise = fluc * randomnumber;

      // return velocity value in x-direction
      if (yplus <= 5.0)
        return utau * yplus + noise;
      else if ((yplus > 5.0) and (yplus <= 30.0))
        return utau * ( 5.0 * log(yplus) - 3.05 ) + noise;
      else if (yplus > 30.0)
      {
        double upre = utau * ( 2.5 * log(yplus) + 5.5 );
        if (upre > umax) return umax + noise;
        else             return upre + noise;
      }
    }
    // return velocity value in z or y-direction
    case 1:
    case 2:
    {
      // generate noise via random number
      randomnumber = DRT::Problem::Instance()->Random()->Uni();
      noise = umax * fluc * randomnumber;

      return noise;
    }
    // nothing to be done for hydrodynamic pressure values
    case 3:
  default:
    return 0.0;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::JefferyHamelFlowFunction::RadialVelocity(
    const double& theta
    )
{
#ifdef DEBUG
  if (theta < -0.01 or theta > (M_PI/4.0+0.01))
  {
    dserror("angle out of range! Must be between 0 and PI/4");
  }
#endif

  const double alpha = theta - (M_PI/8.0);

  // generated by Mathematica 6
  return -170.00000000402173
        + 51.73882280936082*pow(alpha, 2)
        + 1448.6788580912125*pow(alpha, 4)
        + 16137.997882926256*pow(alpha, 6)
        + 93895.51525051043*pow(alpha, 8)
        + 327716.92421506916*pow(alpha,10)
        - 507543.9679454239*pow(alpha,12)
        + 2.5804347181453653e7*pow(alpha,14)
        - 6.684741949638646e8* pow(alpha,16)
        + 1.0642871526439642e10*pow(alpha,18)
        - 1.2944120060793152e11*pow(alpha,20)
        + 1.1321070844277632e12*pow(alpha,22)
        - 7.135693178454414e12* pow(alpha,24)
        + 3.151583189780962e13*pow(alpha,26)
        - 9.234803801165095e13* pow(alpha,28)
        + 1.6246696343829447e14*pow(alpha,30)
        - 1.3098985134542753e14* pow(alpha,32);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double DRT::UTILS::JefferyHamelFlowFunction::Evaluate(int index, const double* xp, double, DRT::Discretization*)
{

  const double x = xp[0];
  const double y = xp[1];

  const double theta = atan(y/x);
  const double u_theta = RadialVelocity(theta);

  // you need to multiply with your viscosity outside of this function to
  // generate flow with higher Reynolds-numbers
  const double nu = 1;

  switch (index)
  {
  case 0:
    return  nu * (u_theta/(x*x+y*y))*x;
  case 1:
    return  nu * (u_theta/(x*x+y*y))*y;
  case 2:
    return 0.0;
  case 3:
    return 0.0;
  default:
    return 0.0;
  }
}


/*----------------------------------------------------------------------*
 | constructor																				    mueller  04/10|
 *----------------------------------------------------------------------*/
DRT::UTILS::WomersleyFunction::WomersleyFunction(bool locsys, int e, int mat, int curve, bool fsi) :
Function(),
isinit_(false),
locsys_(locsys),
locsysid_(e),
radius_(-999.0e99),
tc_(DRT::Problem::Instance()->Curve(curve)),
mat_(mat),
curve_(curve),
viscosity_(-999.0e99),
fsi_(fsi),
locsyscond_(NULL)
{
}
/*----------------------------------------------------------------------*
 |Evaluate Womersley Function 												    mueller 04/10 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::WomersleyFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  /* DESCRIPTION:
   *
   * THIS IS STLL BASIC. DO NOT TRY ANYTHING FANCY...PLEASE!!!!
   *
   * During initialization fluid properties, vessel radius and the local
   * coordinate system (normal, tangent, origin) of the inflow surface are read
   * from the input file. The third direction is built via cross product.
   * In case of a non-circular cross section, the egde nodes of the inflow surface are
   * detected. The node coordinates are transformed to local polar coordinates.
   * The time curve is recorded for one cardiac cycle using the time curve manager.
   * It is then transformed from time to frequency domain by applying a Discrete Fourier
   * Transform (DFT). In order to ensure a smooth transition from pre-stressing velocity
   * to the actual physiological velocity profile during the first second,
   * a linear evolution of the nodevelocity values from w=0 @t=0s to w @ t=1s is set up.
   * After leaving the initialization, the polar coordinates (r,phi) of the current node are
   * determined in case of a non-circular cross section. Its phase is matched to the two
   * closest edge node phase values and the vessel radius R is interpolated.
   * In case of an FSI problem, the edge node coordinates are updated each time step.
   * In case of a circular cross section, the radius is applied and the aforementioned
   * steps are skipped. Now, a vector with time dependent values for each harmonic is
   * generated from the results of the DFT and handed over to the Womersley Function.
   * The results are scaled using the components of the earlier read normal vector and
   * returned depending on index=current global cartesian coordinate (x,y,z)=(0,1,2)
   *
   * what you need in order to run this thing:
   * - DSURF DIRICHLET CONDITION definition(s) for the application of Womersley Dirichlet values;
   *   expample:
   *   //DOBJECT FLAG FLAG FLAG FLAG FLAG FLAG VAL VAL VAL VAL VAL VAL CURVE CURVE CURVE CURVE CURVE CURVE FUNCT FUNCT FUNCT FUNCT FUNCT FUNCT
   *       E 2  -  1    1    1    0    0    0  1.0 1.0 1.0 0.0 0.0 0.0   1     1     1    none  none  none   1     1     1     0     0     0
   *
   *   ( hold also t1-dof and t2-dof because we do not want radial velocities on the inflow surface)
   * - DLINE DIRICHLET CONDITION definition(s) of the line(s) delimiting the Womersley inflow surface;
   *   example:
   *   E 3 - 1 1 1 0 0 0 1.0 1.0 1.0 0.0 0.0 0.0 1 1 1 none none none 1 1 1 0 0 0
   *
   * - a SURF LOCSYS definition like this one:
   *   E 2 - normal nx ny nz tangent tx ty tz origin ox oy oz Type FunctionEvaluation
   *
   * - a CURVE of the type PhysiologicalWaveform
   * - a FUNCT definition like this one:
   *   FUNCT1 WOMERSLEY Local 2 MAT 1 CURVE 1 FSI Yes
   * - in addition: for serial use (NumProc()==1), the origin is calculated anew.
   * 								This might come in handy, when one is to determine the center
   * 								of gravity, i.e. the center line, of the inflow surface.
   * 								(current "usage": you might run it in serial mode first, get the COG-
   * 								coordinates, change your input file accordingly (->LOCSYS) and then rerun
   * 								it in parallel mode. Getting this section to run in parallel mode still causes
   * 								me some headaches. For the time being, just stick to this method!
   * 								Also, for now, please define your inflow surface as ONE Design surface.)
   *
   * further preparations in the FSI DYNAMIC section of your input file
   * - set SHAPEDERIVATIVES to 'no'
   * - select 'iter_monolithicstructuresplit' for COUPALGO
   * - select 'FSIAMG' for LINEARBLOCKSOLVER
   */
  // ONGOING IMPLEMENTATION
  //**********************************************************************
  // Initialization
  //**********************************************************************
  if (!isinit_)
  {
    // get material parameters for fluid
    Teuchos::RCP<MAT::PAR::Material > mat = DRT::Problem::Instance()->Materials()->ById(mat_);
    if (mat->Type() != INPAR::MAT::m_fluid)
      dserror("Material %d is not a fluid",mat_);
    MAT::PAR::Parameter* params = mat->Parameter();
    MAT::PAR::NewtonianFluid* fparams = dynamic_cast<MAT::PAR::NewtonianFluid*>(params);
    if (!fparams)
      dserror("Material does not cast to Newtonian fluid");
    //viscosity_ = fparams->viscosity_;
    // get kinematic viscosity
    viscosity_ = fparams->viscosity_ / fparams->density_;
    fbase_ = 1.0;
    tnminus1_ = -1.0;

    // get local coord system if any
    if (locsys_)
    {
      if (!dis)
        dserror("Have to pass a discretization to allow for local coord systems");
      std::vector<DRT::Condition*> locsys;
      dis->GetCondition("Locsys",locsys);
      if (!locsys.size())
        dserror("No locsys conditions in discretization");
      for (int i=0; i<(int)locsys.size(); ++i)
        if (locsys[i]->Id() == locsysid_)
          locsyscond_ = locsys[i];
      if (!locsyscond_)
        dserror("Cannot find local coord system %d",locsysid_);
      const std::string* type  = locsyscond_->Get<std::string>("Type");
      if (*type!="FunctionEvaluation")
        dserror("Locsys is of wrong type %s",type->c_str());
      const std::vector<double>* n  = locsyscond_->Get<std::vector<double> >("normal");
      const std::vector<double>* t1 = locsyscond_->Get<std::vector<double> >("tangent");
      const std::vector<double>* o  = locsyscond_->Get<std::vector<double> >("origin");
      if (!n || !t1 || !o)
        dserror("LOCSYS Condition components missing");
      normal_   = *n;
      tangent1_ = *t1;
      origin_   = *o;
      tangent2_.resize(tangent1_.size());
      tangent2_[0] = normal_[1]*tangent1_[2]-normal_[2]*tangent1_[1];
      tangent2_[1] = normal_[2]*tangent1_[0]-normal_[0]*tangent1_[2];
      tangent2_[2] = normal_[0]*tangent1_[1]-normal_[1]*tangent1_[0];

      // unit vectors (just in case of non-unit vectors in input file)
      for(int i=0;i<(int)normal_.size();i++)
      {
        normal_[i] /= sqrt(normal_[0]*normal_[0] + normal_[1]*normal_[1] + normal_[2]*normal_[2]);
        tangent1_[i] /= sqrt(tangent1_[0]*tangent1_[0] + tangent1_[1]*tangent1_[1] + tangent1_[2]*tangent1_[2]);
        tangent2_[i] /= sqrt(tangent2_[0]*tangent2_[0] + tangent2_[1]*tangent2_[1] + tangent2_[2]*tangent2_[2]);
      }
      // some output
      std::cout<<"\n"<<"== Womersley Function on surface "<<locsysid_+1;
      if(fsi_)
      	std::cout<<" with FSI-part activated! =="<<std::endl;
      else
      	std::cout<<" with FSI-part disabled! =="<<std::endl;

      //**************************************************************
      // get inflow surface edge nodes
      //**************************************************************
      int linecount = 0;
      const double *tempcoords;
      std::vector<double> xnode;
      std::vector<double> ynode;
      std::vector<double> znode;
      std::vector<DRT::Condition*> dirichlet;

      dis->GetCondition("Dirichlet",dirichlet);
      if (!dirichlet.size())
      dserror("No Dirichlet boundary conditions in discretization");
      // calculation of the centroid, only for serial use
      double centerx = 0.0;
      double centery = 0.0;
      double centerz = 0.0;

      // create node id vector of inflow surface
      for(int i=0;i<(int)dirichlet.size();i++)
      {
        if(dirichlet.at(i)->Id()==locsysid_ && dirichlet.at(i)->Type() == DRT::Condition::SurfaceDirichlet)
        {
          surfnodeids_.assign((int)dirichlet.at(i)->Nodes()->size(), 0);
          for(int j=0;j<(int)dirichlet.at(i)->Nodes()->size();j++)
          surfnodeids_.at(j) = dirichlet.at(i)->Nodes()->at(j);
        }
      }

      // look through all Dirichlet BCs
      for(int i=0;i<(int)dirichlet.size();i++)
      {
        // explicitely look for Line Dirichlet BCs
        if(dirichlet.at(i)->Type() == DRT::Condition::LineDirichlet)
        {
          // check if current Line is part of inflow surface:
          // the lines containing the edge nodes must share all their nodes
          // with the inflow surface. Other line Dirichlet BCs are skipped.
          int nodecount = (int)dirichlet.at(i)->Nodes()->size();

          for(int k=0;k<(int)dirichlet.at(i)->Nodes()->size();k++)
          for(int l=0;l<(int)surfnodeids_.size();l++)
          if(dirichlet.at(i)->Nodes()->at(k)==surfnodeids_.at(l))
          nodecount--;

          if(nodecount==0)
          {
            //cout<<"Line "<<dirichlet.at(i)->Id()<<" lies on surface "<<locsysid_<<endl;
            for(int j=0; j<(int)dirichlet.at(i)->Nodes()->size(); j++)
            {
              int currentid = dirichlet.at(i)->Nodes()->at(j);
              bool havenode = dis->HaveGlobalNode(currentid);
              bool redundant = false;

              nodeids_.push_back(dirichlet.at(i)->Nodes()->at(j));
              // check if node exists on current proc
              if(!havenode) continue;
              // exclude redundant nodes from second node on
              if(j>0)
              for(int k=0;k<j;k++)
              if(currentid==nodeids_.at(k))
              {
                //cout<<"Womersley BC: redundant edge node "<<currentid<<" excluded!"<<endl;
                redundant = true;
                continue;
              }
              if(!redundant)
              {
                tempcoords = dis->gNode(currentid)->X();
                xnode.push_back(tempcoords[0]);
                ynode.push_back(tempcoords[1]);
                znode.push_back(tempcoords[2]);

                // calculation of the centroid (1/2), currently only for serial use
                if(dis->Comm().NumProc()==1)
                {
                  centerx += tempcoords[0];
                  centery += tempcoords[1];
                  centerz += tempcoords[2];
                }
              }
            }
            // counting lines
            linecount++;
          }
          else
          continue;
        }
      }

      if (linecount<1)
      dserror("Define Line Dirichlet BC(s) delimiting the inflow surface in your input file!");
      // calculation of the centroid (2/2), currently only for serial use
      if(dis->Comm().NumProc()==1)
      {
        origin_.at(0) = (centerx /= (double)xnode.size());
        origin_.at(1) = (centery /= (double)ynode.size());
        origin_.at(2) = (centerz /= (double)znode.size());
        std::cout.precision(15);
        std::cout<<"=== newly calculated inflow surface origin: "<<origin_.at(0)<<"   "<<origin_.at(1)<<"   "<<origin_.at(2)<<" ==="<<std::endl;
      }

      //nodal polar coordinates
      std::vector<double> xpedge;
      std::vector<double> xpedgeloc;
      // transform global to local coordinates
      xpedge.assign(3,0.0);
      xpedgeloc.assign(3,0.0);

      for(int i=0; i<(int)xnode.size(); i++)
      {
        xpedge.at(0) = xnode.at(i) - origin_.at(0);
        xpedge.at(1) = ynode.at(i) - origin_.at(1);
        xpedge.at(2) = znode.at(i) - origin_.at(2);
        //(n,t1,t2), signed
        xpedgeloc.assign(3,0.0);
        for(int j=0;j<(int)xpedge.size();j++)
        {
                                  xpedgeloc.at(1) += xpedge.at(j)*tangent1_.at(j);
                                  xpedgeloc.at(2) += xpedge.at(j)*tangent2_.at(j);
        }
        // store r and phi of current node
        phi_.push_back(atan2(xpedgeloc.at(2), xpedgeloc.at(1)));
        noderadius_.push_back(sqrt(xpedge.at(0)*xpedge.at(0) +
																	 xpedge.at(1)*xpedge.at(1) +
                                   xpedge.at(2)*xpedge.at(2)));
      }
      // check for redundant values resulting from having read redundant nodes:
      // the phi-entry in question is not deleted, but simply set to 2pi
      for(int i=0;i<(int)phi_.size();i++)
      for(int j=i+1;j<(int)phi_.size();j++)
      if(phi_.at(i)==phi_.at(j))
      phi_.at(i)=2.0*M_PI;

      // find locations of change of sign
      imaxplus_ = 0;
      imaxminus_ = 0;
      iminplus_ = 0;
      iminminus_ = 0;
      for(int i=0; i<(int)phi_.size(); i++)
      {
        if(fabs(phi_.at(iminminus_))>fabs(phi_.at(i)) && phi_.at(i)<=0.0)
        iminminus_ = i;
        if(fabs(phi_.at(iminplus_))>fabs(phi_.at(i)) && phi_.at(i)>0.0)
        iminplus_ = i;
        if(phi_.at(imaxminus_)>phi_.at(i))
        imaxminus_ = i;
        if(phi_.at(imaxplus_)<phi_.at(i))
        imaxplus_ = i;
      }
    }
    else //if locsys == false
    	dserror("Please check your Womersley Function Definition in your Input file concerning local coordinate systems!");

    //****************************************************************
    // Fourier Analysis of a given time CURVE
    //****************************************************************
    // number of harmonics used for synthesis of a given physiological curve
    noharm_ = 32;
    int sizephyscurve = 200;
    // storage vector for time curve values
    std::vector<double> flowvel;
    // chosen time value from TimeCurveManager
    double timetcm = 0.0;
    // modulus of n-th harmonic during transition phase
    double modtrans;
    // phase between DFT()'s imaginary and real part during transition phase
    double thetatrans;
    // average velocity V(t)
    //physiological curve (>1s)
    for(int j=0; j<sizephyscurve; j++)
    {
      timetcm = 1.0+(double)(j)/(double)(sizephyscurve);
      flowvel.push_back(tc_.f(timetcm));
    }
    // Discrete Fourier Transform (results already scaled with 2/size within DFT())
    DFT(&flowvel, &fouphyscurve_, sizephyscurve);

    //****************************************************************
    // Synthesis of the harmonics vector
    // used for transitional velocity profiles
    //****************************************************************
    // vtemp_ used for transition from parabolic to physiological profile (@t==0)
    for(int l=0;l<(noharm_);l++)
    {
      if(l==0)
      {
        modtrans = sqrt(pow(real(fouphyscurve_.at(l)),2.0) + pow(imag(fouphyscurve_.at(l)),2.0));
        vtemp_.push_back(0.5 * modtrans);
      }
      else
      {
        modtrans = sqrt(real(fouphyscurve_.at(l))*real(fouphyscurve_.at(l)) +
                        imag(fouphyscurve_.at(l))*imag(fouphyscurve_.at(l)));
        thetatrans = -atan2(-imag(fouphyscurve_.at(l)),real(fouphyscurve_.at(l)));
        vtemp_.push_back(modtrans*cos(thetatrans));
      }
    }
    isinit_ = true;
  }// end of initialization

  //**********************************************************************
  // FSI-specific section
  //**********************************************************************
  // if fsi_==true, the edge node coordinates are updated once for
  // each time step applying the same procedure as during initialization
  if(t > tnminus1_)
     dotrafo_ = true;
  if(fsi_ && dotrafo_==true)
  {
    std::vector<double> xnode;
    std::vector<double> ynode;
    std::vector<double> znode;
    std::vector<DRT::Condition*> dirichlet;
    // get current displacement vector from discretization
    RCP<const Epetra_Vector> disp;

    //disp = dis->GetState("displacement");


    dis->GetCondition("Dirichlet",dirichlet);
    if (!dirichlet.size())
      dserror("No Dirichlet boundary conditions in discretization");

    for(int i=0;i<(int)dirichlet.size(); i++)
    if(dirichlet.at(i)->Type() == 2)
    {
      int nodecount = (int)dirichlet.at(i)->Nodes()->size();
      // check if line lies on surface
      for(int k=0;k<(int)dirichlet.at(i)->Nodes()->size();k++)
      for(int l=0;l<(int)surfnodeids_.size();l++)
      if(dirichlet.at(i)->Nodes()->at(k)==surfnodeids_.at(l))
      nodecount--;

      if(nodecount==0)
      {
        for(int j=0; j<(int)dirichlet.at(i)->Nodes()->size(); j++)
        {
          int currentid = dirichlet.at(i)->Nodes()->at(j);
          bool havenode = dis->HaveGlobalNode(currentid);
          bool redundant = false;
          if(!havenode)
          continue;

          if(j>0)
          for(int k=0;k<j;k++)
          if(currentid==nodeids_.at(k))
          {
            redundant = true;
            continue;
          }

          if(!redundant)
          {
            const double *coords = dis->gNode(currentid)->X();
            std::vector<double> tempcoords;
            std::vector<int> dofnode = dis->Dof(dis->gNode(currentid));
            tempcoords.assign(3,0.0);
            // determine current nodal position: reference position + displacement
            for(int k=0; k<3; k++)
            {
              tempcoords.at(k) = coords[k];
              //	if(t>0.0)
              //		tempcoords.at(k) += (*disp)[dis->DofRowMap()->LID( dofnode[k] )];
            }
            xnode.push_back(tempcoords.at(0));
            ynode.push_back(tempcoords.at(1));
            znode.push_back(tempcoords.at(2));
          }
        }
      }
      else
      continue;
    }
    std::vector<double> xpedge;
    std::vector<double> xpedgeloc;

    xpedge.assign(3,0.0);
    xpedgeloc.assign(3,0.0);

    for(int i=0; i<(int)xnode.size(); i++)
    {
      xpedge.at(0) = xnode.at(i) - origin_.at(0);
      xpedge.at(1) = ynode.at(i) - origin_.at(1);
      xpedge.at(2) = znode.at(i) - origin_.at(2);
      xpedgeloc.assign(3,0.0);
      for(int j=0;j<(int)xpedge.size();j++)
      {
        xpedgeloc.at(1) += xpedge.at(j)*tangent1_.at(j);
        xpedgeloc.at(2) += xpedge.at(j)*tangent2_.at(j);
      }
      phi_.at(i) = atan2(xpedgeloc.at(2), xpedgeloc.at(1));
      noderadius_.at(i) = sqrt(xpedge.at(0)*xpedge.at(0) +
															 xpedge.at(1)*xpedge.at(1) +
															 xpedge.at(2)*xpedge.at(2));
    }
    // check for redundant values
    for(int i=0;i<(int)phi_.size();i++)
      for(int j=i+1;j<(int)phi_.size();j++)
        if(phi_.at(i)==phi_.at(j))
        {
          //cout<<"redundant nodal phase value reset to 2*pi to get them out of the picture"<<endl;
          phi_.at(i) = 2.0*M_PI;
        }
    // initialize storage variables
    imaxplus_ = 0;
    imaxminus_ = 0;
    iminplus_ = 0;
    iminminus_ = 0;

    // determine sign switches
    for(int i=0; i<(int)phi_.size(); i++)
    {
      if(fabs(phi_.at(iminminus_))>fabs(phi_.at(i)) && phi_.at(i)<=0.0)
        iminminus_ = i;
      if(fabs(phi_.at(iminplus_))>fabs(phi_.at(i)) && phi_.at(i)>0.0)
        iminplus_ = i;
      if(phi_.at(imaxminus_)>phi_.at(i))
        imaxminus_ = i;
      if(phi_.at(imaxplus_)<phi_.at(i))
        imaxplus_ = i;
    }
    tnminus1_ = t;
    dotrafo_ = false;
  }//end of fsi specific section

  //**********************************************************************
  // Calculation of both current node and edge node radii
  //**********************************************************************
   // distance between (local) origin and given nodal coordinates
  double rabs;
   // current angle
  double phicurr;

  //nodal polar coordinates
  std::vector<double> xptemp;
  std::vector<double> xplocal;
  // transform global to local coordinates
  xptemp.assign(3,0.0);
  xplocal.assign(3,0.0);

  // calculate phase and radius of current node
  for(int i=0;i<(int)xptemp.size();i++)
  xptemp.at(i) = xp[i] - origin_.at(i);
  std::cout.setf(std::ios::scientific);
  std::cout.precision(8);
  //cout<<"xpTemp: "<<xptemp.at(0)<<","<<xptemp.at(1)<<","<<xptemp.at(2)<<endl;
  //(n,t1,t2), signed
  xplocal.assign(3,0.0);

  for(int j=0;j<(int)xptemp.size();j++)
  {
    xplocal.at(0) += xptemp.at(j)*normal_.at(j);
    xplocal.at(1) += xptemp.at(j)*tangent1_.at(j);
    xplocal.at(2) += xptemp.at(j)*tangent2_.at(j);
  }
  // FSI/ALE problem:
  // the ALE mesh gets sucked in or pushed out, hence
  // nodal position changes in normal direction occur.
  // This may lead to erroneous radii, especially when the node's
  // position is close to the local origin.
  // Therefore only the projection of the current node's position
  // into the inflow plane is evaluated, neglecting the normal component.
  // Currently, the reference position is given. Hence, there is no difference.
  // updated node coordinates will be implemented.
  rabs = sqrt(xplocal.at(1)*xplocal.at(1) + xplocal.at(2)*xplocal.at(2));
  phicurr = atan2(xplocal.at(2), xplocal.at(1));
  // determine the two closest edge nodes in terms of phase
  int imin1 = -1;
  int imin2 = -1;
  // closest value to phicurr
  double closest = 100.0;
  // second closest value to phicurr
  double close = 100.0;

  for(int i=0; i<(int)phi_.size(); i++)
  {
    // find location i of closest value phi to phicurr
    if(fabs(phi_.at(i)-phicurr)<closest)
    {
			closest = fabs(phi_.at(i)-phicurr);
			imin1 = i;
    }
  }
  for(int i=0;i<(int)phi_.size();i++)
  {
    // second closest value
    if(fabs(phi_.at(i)-phicurr)<close && i!=imin1)
    {
      close = fabs(phi_.at(i)-phicurr);
      imin2 = i;
    }
    //special cases when phi changes the sign
    if((phicurr>phi_.at(iminminus_) && phicurr<=0.0) || (phicurr<phi_.at(iminplus_) && phicurr>0.0))
    {
      imin1 = iminminus_;
      imin2 = iminplus_;
    }
    if(phicurr<phi_.at(imaxminus_) || phicurr>phi_.at(imaxplus_))
    {
      imin1 = imaxminus_;
      imin2 = imaxplus_;
    }
  }
  //linear interpolation in order to get current approximated vessel radius
  radius_ = noderadius_.at(imin1)+(phicurr-phi_.at(imin1))/(phi_.at(imin2)-phi_.at(imin1))*
    (noderadius_.at(imin2)-noderadius_.at(imin1));


  //**********************************************************************
  // Synthesis of the time-dependant harmonics vector
  // used to calculate the velocity profile
  //**********************************************************************
  // modulus of n-th harmonic during physiological cycle
  double modphys;
  // phase between imaginary and real part during physiological cycle
  double theta;
  // calculation of the harmonic solutions
  std::vector<double> vphyscurve;

  for(int l=0;l<(noharm_);l++)
  {
    if(l==0)
    {
      modphys = sqrt(pow(real(fouphyscurve_.at(l)),2.0) + pow(imag(fouphyscurve_.at(l)),2.0));
      vphyscurve.push_back(0.5 * modphys);
    }
    else
    {
      modphys = sqrt(pow(real(fouphyscurve_.at(l)),2.0) + pow(imag(fouphyscurve_.at(l)),2.0));
      theta = atan2(-imag(fouphyscurve_.at(l)),real(fouphyscurve_.at(l)));
      vphyscurve.push_back(modphys * cos(2.0*M_PI*l*fbase_*(t-1.0) - theta));
    }
  }

  //**********************************************************************
  // Calculation of nodal velocities by components
  //**********************************************************************
  // velocity at given coordinates (Womersley part)
  double w = 0.0;
  // steady part (Hagen-Poiseuille)
  double wsteady = 0.0;
  i_ = std::complex<double> (0.0,1.0);
  // Womersley number
  std::complex<double> alpha(0.0,0.0);
  // z = alpha*i_^(3/2)
  std::complex<double> z(0.0,0.0);
  // term consisting of several Bessel functions
  std::complex<double> bessel(0.0,0.0);

  // calculation of the velocity by components (index)
  // linear transition to physiological profile (may be of advantage to introduce a flexible t_start, here fixed at 1.0)
  if(t<1.0)
  {
    // calculation of the steady part of the solution
    wsteady = 2.0*vtemp_.at(0)/(radius_*radius_)*((radius_*radius_)-(rabs*rabs));
    for(int k=1;k<noharm_;k++)
    {
      // Womersley number
      alpha = radius_*sqrt(2.0*M_PI*k*fbase_/viscosity_);
      z = alpha*std::pow(i_,1.5);
      // Bessel term
      bessel = z*(BesselJ01(z,false)-BesselJ01(z*(std::complex<double>)(rabs/radius_), false))/
        (z*BesselJ01(z,false)-(std::complex<double>)(2.0)*BesselJ01(z, true));
      w += vtemp_.at(k)*real(bessel);
    }
    // division through time curve value because of result = VALUE*CURVE*FUNCT
    if(tc_.f(t)==0.0)
       return 0.0;
    else
    {
      w = (w + wsteady)/tc_.f(t)*t;
      // calculate normal component (in opposite direction due to the normal being an outward normal)
      w *= -normal_[index];
      return w;
    }
  }
  // physiological solution
  if(t>=1.0)
  {
    wsteady = 2.0*vphyscurve.at(0)/(radius_*radius_)*((radius_*radius_)-(rabs*rabs));
    for(int k=1;k<noharm_;k++)
    {
      alpha = radius_*sqrt(2.0*M_PI*k*fbase_/viscosity_);
      z = alpha*pow(i_,1.5);
      bessel= z*(BesselJ01(z,false)-BesselJ01(z*(std::complex<double>)(rabs/radius_), false))/
        (z*BesselJ01(z,false)-(std::complex<double>)(2.0)*BesselJ01(z, true));
      w += vphyscurve.at(k)*real(bessel);
    }
    if(tc_.f(t)==0.0)
    w = (w + wsteady)/tcprevious_;
    else
    {
      w = (w + wsteady)/tc_.f(t);
      tcprevious_ = tc_.f(t);
    }
    w *= -normal_[index];
    return w;
  }

  return -999.0e99;
}
/*----------------------------------------------------------------------*
 |  Womersley: Bessel functions of order 0 and 1          mueller 04/10 |
 *----------------------------------------------------------------------*/
std::complex<double> DRT::UTILS::WomersleyFunction::BesselJ01(std::complex<double> z, bool order)
{
  // DESCRIPTION:
  // Bessel functions of order 0 (order==false) or 1 (order==true) are calculated for
  // a given argument z

  int end = 70;
  std::complex<double> J(0.0,0.0);
  double fac = 1.0;

  // Bessel function of the first kind and order 0
  if(order==false)
  {
    for(int m=0;m<end;m++)
    {
      for(int k=2;k<=m;k++)
	fac *= (double)(k);
      J += (std::complex<double>)((double)(pow(-1.0,(double)(m)))/pow(fac,2.0))*
      pow((z/(std::complex<double>)(2.0)),(std::complex<double>)(2*m));
      fac = 1.0;
    }
    if(z == std::complex<double>(0.0,0.0))
      J= std::complex<double>(1.0,0.0);
  }
  // Bessel function of the first kind and order 1
  else
  {
    for(int m=0;m<end;m++)
    {
      for(int k=2;k<=m;k++)
	fac *= (double)(k);
      J += (std::complex<double>)((pow(-1.0,(double)(m)))/((double)(m+1)*pow(fac,2.0)))*
      pow((z/std::complex<double>(2.0)),(std::complex<double>)(2*m+1));
      fac = 1.0;
    }
  }
  return J;
}
/*----------------------------------------------------------------------*
 |  Womersley: Discrete Fourier Transfomation             mueller 04/10 |
 *----------------------------------------------------------------------*/
void DRT::UTILS::WomersleyFunction::DFT(std::vector<double> *data, std::vector< std::complex<double> > *resdata, const int N)
{
  // DESCRIPTION:
  // a given vector data undergoes the DFT. The result is written back to resdata. N is the number
  // of sampling points

  std::complex<double> exp_w;
  std::complex<double> c;
  std::complex<double> sum;
  c = std::complex<double>(2.0/(double)(N), 0.0);

  for(int k=0;k<N;k++)
  {
    sum = std::complex<double>(0.0,0.0);
    for(int n=0;n<N;n++)
    {
      exp_w= std::complex<double>(cos(2.0*M_PI/double(N)*k*n), -sin(2.0*M_PI/double(N)*k*n));
      sum += c*data->at(n)*exp_w;
    }
    resdata->push_back(sum);
  }
  return;
}
/*----------------------------------------------------------------------*
 | constructor                                              henke 05/09 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ZalesaksDiskFunction::ZalesaksDiskFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"  schott 06/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::ZalesaksDiskFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  // the disk consists of 3 lines and a part of a circle and four points
  // decide if the orthogonal projection of the current point lies on the lines and the circle (four different distances possible)
  // additionally the smallest distance can be between the current point and one of the four corners

  double distance = 99999.0;

  //=====================================
  // distances to the four corners
  //=====================================
  // upper points
  double y_upper = sqrt(DSQR(0.15)-DSQR(0.025)) + 0.25;
  // warning: sign must be positive
  double dist_lu = sqrt(DSQR(xp[0]+0.025)+DSQR(xp[1]-y_upper));
  if(abs(dist_lu) < abs(distance)) distance = dist_lu;

  // warning: sign must be positive
  double dist_ru = sqrt(DSQR(xp[0]-0.025)+DSQR(xp[1]-y_upper));
  if(abs(dist_ru) < abs(distance)) distance = dist_ru;

  // under points
  double y_down = 0.15;
  // warning: sign must be negative
  double dist_ld = sqrt(DSQR(xp[0]+0.025)+DSQR(xp[1]-y_down));
  if(abs(dist_ld) < abs(distance)) distance = -dist_ld;

  // warning: sign must be negative
  double dist_rd = sqrt(DSQR(xp[0]-0.025)+DSQR(xp[1]-y_down));
  if(abs(dist_rd) < abs(distance)) distance = -dist_rd;

  //=====================================
  // projection on the 3 lines
  //=====================================
  // decide for vertical lines
  if(xp[1]>= 0.15 && xp[1]<= y_upper)
  {
    // leftVertLine
    if(abs(xp[0]+0.025) < abs(distance)) distance = xp[0]+0.025;

    // rightVertLine
    if(abs(0.025-xp[0]) < abs(distance)) distance = 0.025-xp[0];
  }
  // decide for horizontal line
  if(xp[0]>= -0.025 && xp[0]<=0.025)
  {
    // horizontalLine
    if(abs(xp[1]-0.15) < abs(distance)) distance = xp[1]-0.15;
  }

  //======================================
  // distance to the circle
  //======================================
  // decide for part of circle
  // get radius of sphere for current point
  double s = sqrt(DSQR(xp[0]-0.0)+DSQR(xp[1]-0.25));
  // get angle between line form midpoint of circle to current point and vector (0,1,0)
  double y_tmp= sqrt(DSQR(0.15)-DSQR(0.025))*s/0.15;
  if((xp[1]-0.25) <= y_tmp)
  {
    if(abs(s-0.15) < abs(distance)) distance = s-0.15;
  }

  return distance;
}

/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame2Function::CircularFlame2Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame2Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double dens_minus = 1.0;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // -pres + dynvisc*2*du_x/dx
  double flux = 0.5*dens_minus*radius*radius*u*u/(xp[0]*xp[0]+xp[1]*xp[1])
              + 2.0*dens_minus*sl*u*log(sqrt(xp[0]*xp[0]+xp[1]*xp[1]))
              + visc_minus*2.0*radius*u/(xp[0]*xp[0]+xp[1]*xp[1])*(1.0-2.0*xp[0]*xp[0]/(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}

/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame3Function::CircularFlame3Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame3Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double dens_minus = 1.0;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // -pres + dynvisc*2*du_y/dy
  double flux = 0.5*dens_minus*radius*radius*u*u/(xp[0]*xp[0]+xp[1]*xp[1])
              + 2.0*dens_minus*sl*u*log(sqrt(xp[0]*xp[0]+xp[1]*xp[1]))
              + visc_minus*2.0*radius*u/(xp[0]*xp[0]+xp[1]*xp[1])*(1.0-2.0*xp[1]*xp[1]/(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}

/*----------------------------------------------------------------------*
 | constructor                                              henke 01/12 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CircularFlame4Function::CircularFlame4Function() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of circular flame test function henke               01/12 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CircularFlame4Function::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  const double visc_minus = 0.001;
  const double radius_0 = 0.025;
  const double u = 1.0;
  const double sl = 1.0;
  double radius = radius_0 + t*(u+sl);

  // dynvisc*2*du_x/dy
  double flux = -2.0*visc_minus*radius*u*2.0*xp[0]*xp[1]/((xp[0]*xp[0]+xp[1]*xp[1])*(xp[0]*xp[0]+xp[1]*xp[1]));

  return flux;
}

/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 04/10 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CollapsingWaterColumnFunction::CollapsingWaterColumnFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of two-phase flow test case               rasthofer 04/10 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CollapsingWaterColumnFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of distance (sign is already taken in consideration)
  double distance = 0.0;

//  if (xp[0] > -0.146 and xp[1] > 0.067)
//  {
//     distance = + sqrt(DSQR(-0.146 - xp[0])+DSQR(0.067 - xp[1]));
//  }
//  else if (xp[0] >= -0.146 and xp[1] <= 0.067)
//  {
//     distance = + fabs(-0.146 - xp[0]);
//     //std::cout << distance << std::endl;
//  }
//  else if (xp[0] <= -0.146 and xp[1] >= 0.067)
//  {
//     distance = + fabs(0.067 - xp[1]);
//  }
//  else if (xp[0] < -0.146 and xp[1] < 0.067 and xp[1] <= (xp[0] + 0.213))
//  {
//     //distance = - fabs(0.067 - xp[1]);
//     distance = - fabs(-0.146 - xp[0]);
//  }
//  else
//  {
//     //distance = - fabs(-0.146 - xp[0]);
//     distance = - fabs(0.067 - xp[1]);
//     //std::cout << distance << std::endl;
//  }

  double xp_corner[2];
  double xp_center[2];
  double radius=0.03;

  xp_corner[0]=0.146;
  xp_corner[1]=0.292;
  xp_center[0]=xp_corner[0]-radius;
  xp_center[1]=xp_corner[1]-radius;


  if (xp[0] <=xp_center[0] and xp[1] >= xp_center[1])
  {
     distance = xp[1]-xp_corner[1] ;
  }
  else if (xp[0] >=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  {
      distance= xp[0]-xp_corner[0];
  }
  else if (xp[0] <xp_center[0] and xp[1] < xp_center[1])
  {
      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
      {
          distance = - fabs(xp_corner[1] - xp[1]);
      }
      else
      {
          distance = - fabs(xp_corner[0] - xp[0]);
      }
  }
  else
  {
      distance = sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                          rasthofer 04/10 |
 *----------------------------------------------------------------------*/
DRT::UTILS::CollapsingWaterColumnFunctionCoarse::CollapsingWaterColumnFunctionCoarse() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of two-phase flow test case               rasthofer 04/10 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::CollapsingWaterColumnFunctionCoarse::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  double xp_corner[2];
  double xp_center[2];
    double distance = 0.0;
  double radius=0.03;

  //xp_corner[0]=0.146;//0.06
  xp_corner[0]=0.06;//0.06
  //xp_corner[1]=0.292;//0.06
  xp_corner[1]=0.06;//0.06

  xp_center[0]=xp_corner[0]-radius;
  xp_center[1]=xp_corner[1]-radius;


  if (xp[0] <=xp_center[0] and xp[1] >= xp_center[1])
  {
     distance = xp[1]-xp_corner[1] ;
  }
  else if (xp[0] >=xp_center[0] and xp[1] <= xp_center[1] and !(xp[0]==xp_center[0] and xp[1]==xp_center[1]))
  {
      distance= xp[0]-xp_corner[0];
  }
  else if (xp[0] <xp_center[0] and xp[1] < xp_center[1])
  {
      if(xp[1]>(xp_corner[1]+(xp[0]-xp_corner[0])))
      {
          distance = - fabs(xp_corner[1] - xp[1]);
      }
      else
      {
          distance = - fabs(xp_corner[0] - xp[0]);
      }
  }
  else
  {
      distance = sqrt(DSQR(xp[0]-xp_center[0])+DSQR(xp[1]-xp_center[1]))-radius;
  }

  return distance;
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 10/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ORACLESGFunction::ORACLESGFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"   henke 05/09 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::ORACLESGFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{

  if (xp[0] > 0.0)
    dserror("invalid coordinate for ORACLES G-function function!");

  const double eps = 0.00152;
  //const double zsing = 0.7525-0.05;//0.0354;

  double gfuncvalue = 0.0;

  // implementation for periodic spanwise boundary
  if (xp[1] >= 0.0)
    gfuncvalue = (xp[1]-0.0354) - eps;
  else
    gfuncvalue = (-0.0354-xp[1]) - eps;

#if 0
  // implementation for spanwise walls
  if ( xp[2] <= -zsing and abs(xp[1]) <= abs(xp[2]+zsing) )
  {
    gfuncvalue = (-0.7525-xp[2]) - eps;
  }
  else if ( xp[2] >= zsing and abs(xp[1]) <= (xp[2]-zsing) )
  {
    gfuncvalue = (xp[2]-0.7525) - eps;
  }
  else if ( xp[1] >= 0.0 and ( xp[1] > abs(xp[2]+zsing) or xp[1] > (xp[2]-zsing) ))
  {
    gfuncvalue = (xp[1]-0.0354) - eps;
  }
  else if ( xp[1] < 0.0 and (-xp[1] > abs(xp[2]+zsing) or -xp[1] > (xp[2]-zsing) ))
  {
    gfuncvalue = (-0.0354-xp[1]) - eps;
  }
  else
    dserror("coordinate out of range of ORACLES G-function function");
#endif

  return gfuncvalue;
}

/*----------------------------------------------------------------------*
 | constructor                                             schott 05/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::RotatingConeFunction::RotatingConeFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Rotating Cone "  schott 05/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::RotatingConeFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of distance (sign is already taken in consideration)
  double distance = 0;


  double x0c = 1.0/6.0;
  double x1c = 1.0/6.0;

  double sigma = 0.2;

  double X0 = (xp[0]-x0c)/sigma;
  double X1 = (xp[1]-x1c)/sigma;

  double radius = sqrt(DSQR(X0)+DSQR(X1));

  if(radius <= 1.0) distance = 0.25*(1.0+cos(PI*X0))*(1.0+cos(PI*X1));
  else distance = 0.0;


  return (distance-1.0);
}


/*----------------------------------------------------------------------*
 | constructor                                              henke 05/11 |
 *----------------------------------------------------------------------*/
DRT::UTILS::LevelSetCutTestFunction::LevelSetCutTestFunction() :
Function()
{
}

/*----------------------------------------------------------------------*
 | evaluation of level set test function "Zalesak's disk"   henke 05/11 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::LevelSetCutTestFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
  //here calculation of phi (sign is already taken in consideration)
  double phi = 0;

  // column of nodes (x < -0.7)
  if (xp[0] < -0.75)
    phi = xp[0]+0.6;
  // column of nodes (x = -0.7)
  else if ((xp[0] > -0.75) and (xp[0] < -0.65))
    phi = -0.1;
  // column of nodes (x = -0.6)
  else if ((xp[0] > -0.65) and (xp[0] < -0.55))
    phi = 0.0 +1.0E-4;
  // column of nodes (x = -0.5)
  else if ((xp[0] > -0.55) and (xp[0] < -0.45))
    phi = 0.1;
  // column of nodes (x = -0.4)
  else if ((xp[0] > -0.45) and (xp[0] < -0.35))
    phi = 0.0 +1.0E-5;
  // column of nodes (x = -0.3)
  else if ((xp[0] > -0.35) and (xp[0] < -0.25))
    phi = -0.1;
  // column of nodes (x = -0.2)
  else if ((xp[0] > -0.25) and (xp[0] < -0.15))
    phi = 0.0 +1.0E-12;
  // column of nodes (x = -0.1)
  else if ((xp[0] > -0.15) and (xp[0] < -0.05))
    phi = 0.1;
  // column of nodes (x = 0.0)
  else if ((xp[0] > -0.05) and (xp[0] < 0.05))
    phi = 0.0;
  // column of nodes (x = 0.1)
  else if ((xp[0] > 0.05) and (xp[0] < 0.15))
    phi = -0.1;
  // column of nodes (x = 0.2)
  else if ((xp[0] > 0.15) and (xp[0] < 0.25))
    phi = 0.0 -1.0E-12;
  // column of nodes (x = 0.3)
  else if ((xp[0] > 0.25) and (xp[0] < 0.35))
    phi = 0.1;
  // column of nodes (x = 0.4)
  else if ((xp[0] > 0.35) and (xp[0] < 0.45))
    phi = 0.0 -1.0E-5;
  // column of nodes (x = 0.5)
  else if ((xp[0] > 0.45) and (xp[0] < 0.55))
    phi = -0.1;
  // column of nodes (x = 0.6)
  else if ((xp[0] > 0.55) and (xp[0] < 0.65))
    phi = 0.0 +xp[1]*0.0001;
  // column of nodes (x = 0.7)
  else if ((xp[0] > 0.65) and (xp[0] < 0.75))
    phi = 0.1;
  // column of nodes (x = 0.8)
  else if ((xp[0] > 0.75) and (xp[0] < 0.85))
    phi = 0.0 -(xp[1]-0.001)*0.0001;
  // column of nodes (x = 0.9)
  else if ((xp[0] > 0.85) and (xp[0] < 0.95))
    phi = -0.1;
  // column of nodes (x = 1.0)
  else if (xp[0] > 0.95)
    phi = -0.2;
  // something went wrong
  else
   dserror("this node does not exist");
  return phi;
}

/*----------------------------------------------------------------------*
 | Constructor of ControlledRotation                         hahn 04/13 |
 *----------------------------------------------------------------------*/
DRT::UTILS::ControlledRotationFunction::ControlledRotationFunction(std::string fileName, std::string type, double origin_x, double origin_y, double origin_z) :
Function(), NUMMANEUVERCELLS_(4)
{
    // Initialize variables
    // *****************************************************************************

    // Initialize condition type
    if (type == "STRUCTURE") {    // structure
        type_ = 1;
    } else if (type == "FLUID") { // fluid
        type_ = 2;
    } else {
        dserror("When using the function CONTROLLEDROTATION, the type must be either 'STRUCTURE' or 'FLUID'");
    }

    // Initialize origin, about which the rotation is performed
    origin_(0,0) = origin_x;
    origin_(1,0) = origin_y;
    origin_(2,0) = origin_z;

    // Initialize time of previous time step (at t-deltaT)
    timeOld_ = 0.0;

    // Initialize previous angular acceleration (at t-deltaT)
    omegaDotOld_B_(0,0) = 0;
    omegaDotOld_B_(1,0) = 0;
    omegaDotOld_B_(2,0) = 0;

    // Current angular rate (at t)
    omega_B_(0,0) = 0;
    omega_B_(1,0) = 0;
    omega_B_(2,0) = 0;

    // Initialize satellite's current attitude trafo matrix from B- to I-system (at t)
    satAtt_dcm_IB_(0,0) = 1.0; satAtt_dcm_IB_(0,1) = 0.0; satAtt_dcm_IB_(0,2) = 0.0;
    satAtt_dcm_IB_(1,0) = 0.0; satAtt_dcm_IB_(1,1) = 1.0; satAtt_dcm_IB_(1,2) = 0.0;
    satAtt_dcm_IB_(2,0) = 0.0; satAtt_dcm_IB_(2,1) = 0.0; satAtt_dcm_IB_(2,2) = 1.0;

    // Initialize satellite's current attitude quaternion from B- to I-system (at t)
    satAtt_q_IB_(0,0) = 0;
    satAtt_q_IB_(1,0) = 0;
    satAtt_q_IB_(2,0) = 0;
    satAtt_q_IB_(3,0) = 1;

    // Initialize maneuvers
    maneuvers_.clear();

    // Read maneuver file and fill maneuvers variable
    // *****************************************************************************

    std::string line;
    std::stringstream lineStream;
    std::string cell;

    // Open file
    std::ifstream file (fileName.c_str());
    if (!file.is_open()) {
        dserror("Unable to open file: %s", fileName.c_str());
    }

    // Loop through all lines
    while (getline (file,line)) {
        if (!line.empty()) {
            // Clear local variables
            lineStream.clear();
            cell.clear();

            // Obtain all numManeuverCells=4 values from current line (t, omegaDot_x_B, omegaDot_y_B, omegaDot_z_B)
            lineStream << line;
            for (int i=0; i<NUMMANEUVERCELLS_; i++) {
                // Obtain i-th cell from current line
                getline(lineStream, cell, ' ');

                // If empty cell, than either empty line or one cell in the line
                // missing, anyhow an error.
                if (cell.empty()) {
                    dserror("Error during reading of file: %s", fileName.c_str());
                }

                // Convert input cell from string to double
                double cellDouble = (double)strtod(cell.c_str(), NULL);

                // Add cell to maneuvers vector
                maneuvers_.push_back(cellDouble);
            }
        }
    }

    // Close file
    file.close();

    // Determine number of maneuvers
    numManeuvers_ = (int)(maneuvers_.size()) / (int)NUMMANEUVERCELLS_;

    // Output maneuver list
    printf("\n=================================================================================\n");
    printf("ControlledRotation - %s\n", type.c_str());
    printf("---------------------------------------------------------------------------------\n");
    printf("The following %d maneuvers have been loaded from file %s:\n", numManeuvers_, fileName.c_str());
    for (int i=0; i<numManeuvers_; i++) {
        printf("Time: %e  OmegaDot_B: %e, %e, %e \n",
                maneuvers_[i*NUMMANEUVERCELLS_+0], maneuvers_[i*NUMMANEUVERCELLS_+1],
                maneuvers_[i*NUMMANEUVERCELLS_+2], maneuvers_[i*NUMMANEUVERCELLS_+3]);
    }
    printf("=================================================================================\n\n");

}

/*----------------------------------------------------------------------*
 | Evaluate ControlledRotation and return for structures the current    |
 | displacement and for fluids the current velocity of the respective   |
 | node for the given index.                                 hahn 04/13 |
 *----------------------------------------------------------------------*/
double DRT::UTILS::ControlledRotationFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
    // Check, if a restart has been performed
    // *****************************************************************************
    const int step = DRT::Problem::Instance()->Restart();
    if ((step > 0) && (timeOld_ == 0.0)) {

      if (type_ != 1) { // fluid
        dserror("When using the function CONTROLLEDROTATION, the restart functionality only works for structures");
      }

      Teuchos::RCP<DRT::Discretization> rcpdis = Teuchos::rcp(dis, false);
      IO::DiscretizationReader reader(rcpdis,step-1);

      // Set time from last completed time step
      timeOld_ = reader.ReadDouble("timeOld");

      // Set angular rate from last completed time step
      omega_B_(0,0) = reader.ReadDouble("omega_B_x");
      omega_B_(1,0) = reader.ReadDouble("omega_B_y");
      omega_B_(2,0) = reader.ReadDouble("omega_B_z");

      // Set angular acceleration from next-to-last completed time step
      omegaDotOld_B_(0,0) = reader.ReadDouble("omegaDotOld_B_x");
      omegaDotOld_B_(1,0) = reader.ReadDouble("omegaDotOld_B_y");
      omegaDotOld_B_(2,0) = reader.ReadDouble("omegaDotOld_B_z");

      // Set attitude quaternion from last completed time step
      satAtt_q_IB_(0,0) = reader.ReadDouble("satAtt_q_IB_1");
      satAtt_q_IB_(1,0) = reader.ReadDouble("satAtt_q_IB_2");
      satAtt_q_IB_(2,0) = reader.ReadDouble("satAtt_q_IB_3");
      satAtt_q_IB_(3,0) = reader.ReadDouble("satAtt_q_IB_4");
    }

    // If new time step, apply angular acceleration (if desired) and determine
    // new attitude satAtt_dcm_IB_
    // *****************************************************************************
    // Determine time difference
    double deltaT = t - timeOld_;

    if (deltaT > 1e-12) { // new time step

        // Determine current angular acceleration (at t)
        // -----------------------------------------------------------------------------
        LINALG::Matrix<3,1> omegaDot_B;
        omegaDot_B(0,0) = 0.0;
        omegaDot_B(1,0) = 0.0;
        omegaDot_B(2,0) = 0.0;

        for (int i=0; i<numManeuvers_; i++) {
            if (t >= maneuvers_[i*NUMMANEUVERCELLS_+0]) {
                omegaDot_B(0,0) = maneuvers_[i*NUMMANEUVERCELLS_+1];
                omegaDot_B(1,0) = maneuvers_[i*NUMMANEUVERCELLS_+2];
                omegaDot_B(2,0) = maneuvers_[i*NUMMANEUVERCELLS_+3];
            }
        }

        // Calculate current angular rate (at t) by integration
        // of angular acceleration (trapezoidal rule):
        // omega_(t) = deltaT * (omegaDotOld + omegaDot) / 2 + omega_(t-deltaT)
        // -----------------------------------------------------------------------------
        LINALG::Matrix<3,1> deltaOmega;
        deltaOmega.Update(omegaDotOld_B_, omegaDot_B); // 1) deltaOmega <- omegaDotOld_ + omegaDot
        deltaOmega.Scale(deltaT/2.0);                  // 2) deltaOmega <- deltaOmega * deltaT / 2.0
        omega_B_ += deltaOmega;                        // 3) omega_ <- omega_ + deltaOmega

        /* // Debugging output
           cout << "omegaDot: "; omegaDot_B.Print(cout); // Print omegaDot_B
           cout << "omega:    "; omega_B_.Print(cout);   // Print omega_B_
        */

        omegaDotOld_B_ = omegaDot_B; // Set omegaDotOld_B_ for next time step

        // Calculate new attitude quaternion satAtt_q_IB_ [Wertz, p. 511f]
        // -----------------------------------------------------------------------------
        LINALG::Matrix<4,4> mOmega; // Skew-symmetric matrix containing angular velocity components
        mOmega(0,0) =  0.0;           mOmega(0,1) =  omega_B_(2,0); mOmega(0,2) = -omega_B_(1,0); mOmega(0,3) = omega_B_(0,0);
        mOmega(1,0) = -omega_B_(2,0); mOmega(1,1) =  0.0;           mOmega(1,2) =  omega_B_(0,0); mOmega(1,3) = omega_B_(1,0);
        mOmega(2,0) =  omega_B_(1,0); mOmega(2,1) = -omega_B_(0,0); mOmega(2,2) =  0.0;           mOmega(2,3) = omega_B_(2,0);
        mOmega(3,0) = -omega_B_(0,0); mOmega(3,1) = -omega_B_(1,0); mOmega(3,2) = -omega_B_(2,0); mOmega(3,3) = 0.0;

        mOmega.Scale(deltaT/2.0);
        mOmega(0,0) = 1.0;
        mOmega(1,1) = 1.0;
        mOmega(2,2) = 1.0;
        mOmega(3,3) = 1.0;

        LINALG::Matrix<4,1> satAtt_q_IB_TMP;
        satAtt_q_IB_TMP.Multiply(mOmega, satAtt_q_IB_);
        satAtt_q_IB_ = satAtt_q_IB_TMP;

        satAtt_q_IB_.Scale(1/satAtt_q_IB_.Norm2()); // Normalize attitude quaternion

        // Create transformation matrix satAtt_dcm_IB_ [Wertz, (E-8)]
        // -----------------------------------------------------------------------------
        const double q1 = satAtt_q_IB_(0,0);
        const double q2 = satAtt_q_IB_(1,0);
        const double q3 = satAtt_q_IB_(2,0);
        const double q4 = satAtt_q_IB_(3,0);

        satAtt_dcm_IB_(0,0) = q1*q1-q2*q2-q3*q3+q4*q4; satAtt_dcm_IB_(0,1) = 2.0*(q1*q2-q3*q4);        satAtt_dcm_IB_(0,2) = 2.0*(q1*q3+q2*q4);
        satAtt_dcm_IB_(1,0) = 2.0*(q1*q2+q3*q4);       satAtt_dcm_IB_(1,1) = -q1*q1+q2*q2-q3*q3+q4*q4; satAtt_dcm_IB_(1,2) = 2.0*(q2*q3-q1*q4);
        satAtt_dcm_IB_(2,0) = 2.0*(q1*q3-q2*q4);       satAtt_dcm_IB_(2,1) = 2.0*(q2*q3+q1*q4);        satAtt_dcm_IB_(2,2) = -q1*q1-q2*q2+q3*q3+q4*q4;

        // Update time of last time step
        // -----------------------------------------------------------------------------
        timeOld_ = t;

        // Write restart information
        // -----------------------------------------------------------------------------
        Teuchos::RCP<DRT::Discretization> rcpdis = Teuchos::rcp(dis, false);
        Teuchos::RCP<IO::DiscretizationWriter> output = Teuchos::rcp(new IO::DiscretizationWriter(rcpdis));

        // Write time
        // Note: this is needed due to the fact that the restart information of the current
        // time step X is written to the time step X-1 in the control file. This is due to
        // the fact that output->NewStep() is only called after this function by the
        // corresponding structure or fluid function.
        output->WriteDouble("timeOld", timeOld_);

        // Write angular rate
        output->WriteDouble("omega_B_x", omega_B_(0,0));
        output->WriteDouble("omega_B_y", omega_B_(1,0));
        output->WriteDouble("omega_B_z", omega_B_(2,0));

        // Write previous angular acceleration
        output->WriteDouble("omegaDotOld_B_x", omegaDotOld_B_(0,0));
        output->WriteDouble("omegaDotOld_B_y", omegaDotOld_B_(1,0));
        output->WriteDouble("omegaDotOld_B_z", omegaDotOld_B_(2,0));

        // Write attitude quaternion
        output->WriteDouble("satAtt_q_IB_1", satAtt_q_IB_(0,0));
        output->WriteDouble("satAtt_q_IB_2", satAtt_q_IB_(1,0));
        output->WriteDouble("satAtt_q_IB_3", satAtt_q_IB_(2,0));
        output->WriteDouble("satAtt_q_IB_4", satAtt_q_IB_(3,0));
    }

    // Obtain the current node position in the inertial system
    // *****************************************************************************
    // NOTE: 1) Here it is assumed that the difference between the mesh system and the
    //          body system is just given by a constant displacement named origin_.
    //          Hence the variable origin_ specifies the point, given in the mesh
    //          system, about which is being rotated.
    //       2) The inertial system used here has position and attitude of the body
    //          system at initial time. Thus it is deviating from the ECI system J2000
    //          at most by a constant displacement that is due to the satellite position
    //          at initial time and a constant rotation that is due to the satellite's
    //          attitude at initial time. Due to Galilei-invariance this shouldn't be
    //          a problem and it can easily be converted to the J2000 system.

    // Node reference position, given in the body system
    LINALG::Matrix<3,1> nodeReferencePos_B;
    nodeReferencePos_B(0,0) = xp[0] - origin_(0,0);
    nodeReferencePos_B(1,0) = xp[1] - origin_(1,0);
    nodeReferencePos_B(2,0) = xp[2] - origin_(2,0);

    // Node position, given in the inertial system
    LINALG::Matrix<3,1> nodePos_I;
    nodePos_I.Multiply(satAtt_dcm_IB_, nodeReferencePos_B);

    // Calculate and return the displacement/velocity of the node for the given index
    // *****************************************************************************

    if (t<=0) {
        // Return zero displacement/translational velocity of the node for the given index
        return 0.0;
    }

    // Displacement of the node for the given index
    const double dispNodePos_I = nodePos_I(index,0) - nodeReferencePos_B(index,0);

    if (type_ == 1) { // structure

        // Return the displacement of the node for the given index
        return dispNodePos_I;

    } else if (type_ == 2) { // fluid

        // Node velocity, given in the inertial system: v = omega x r
        double nodeVel_I[3];
        nodeVel_I[0] = omega_B_(1,0) * nodePos_I(2,0) - omega_B_(2,0) * nodePos_I(1,0);
        nodeVel_I[1] = omega_B_(2,0) * nodePos_I(0,0) - omega_B_(0,0) * nodePos_I(2,0);
        nodeVel_I[2] = omega_B_(0,0) * nodePos_I(1,0) - omega_B_(1,0) * nodePos_I(0,0);

        // Return the translational velocity of the node for the given index
        return (nodeVel_I[index]);

    } else {
        dserror("When using the function CONTROLLEDROTATION, the type must be either 'STRUCTURE' or 'FLUID'");
        return 0.0;
    }
}

/*----------------------------------------------------------------------*
 | Constructor of AccelerationProfile                        hahn 09/13 |
 *----------------------------------------------------------------------*/
DRT::UTILS::AccelerationProfileFunction::AccelerationProfileFunction(std::string fileName) :
Function(), NUMACCELERATIONCELLS_(4)
{
    // Initialize variables
    // *****************************************************************************

    // Initialize time of previous time step (at t-deltaT)
    timeOld_ = 0.0;

    // Initialize accelerations
    accelerations_.clear();

    // Initialize current acceleration (at t)
    acc_B_(0,0) = 0.0;
    acc_B_(1,0) = 0.0;
    acc_B_(2,0) = 0.0;

    // Read acceleration profile file and fill acceleration variable
    // *****************************************************************************

    std::string line;
    std::stringstream lineStream;
    std::string cell;

    // Open file
    std::ifstream file (fileName.c_str());
    if (!file.is_open()) {
        dserror("Unable to open file: %s", fileName.c_str());
    }

    // Loop through all lines
    while (getline (file,line)) {
        if (!line.empty()) {
            // Clear local variables
            lineStream.clear();
            cell.clear();

            // Obtain all numAccelerationCells=4 values from current line (t, acc_x_B, acc_y_B, acc_z_B)
            lineStream << line;
            for (int i=0; i<NUMACCELERATIONCELLS_; i++) {
                // Obtain i-th cell from current line
                getline(lineStream, cell, ' ');

                // If empty cell, than either empty line or one cell in the line
                // missing, anyhow an error.
                if (cell.empty()) {
                    dserror("Error during reading of file: %s", fileName.c_str());
                }

                // Convert input cell from string to double
                double cellDouble = (double)strtod(cell.c_str(), NULL);

                // Add cell to acceleration vector
                accelerations_.push_back(cellDouble);
            }
        }
    }

    // Close file
    file.close();

    // Determine number of accelerations
    numAccelerations_ = (int)(accelerations_.size()) / (int)NUMACCELERATIONCELLS_;

    // Output acceleration list
    printf("\n=================================================================================\n");
    printf("AccelerationProfile\n");
    printf("---------------------------------------\n");
    printf("The following %d acceleration rows have been loaded from file %s:\n", numAccelerations_, fileName.c_str());
    for (int i=0; i<numAccelerations_; i++) {
        printf("Time: %e  acc_B: %e, %e, %e \n",
                accelerations_[i*NUMACCELERATIONCELLS_+0], accelerations_[i*NUMACCELERATIONCELLS_+1],
                accelerations_[i*NUMACCELERATIONCELLS_+2], accelerations_[i*NUMACCELERATIONCELLS_+3]);
    }
    printf("=================================================================================\n\n");

}

/*---------------------------------------------------------------------*
 | Evaluate AccelerationProfile and return the respective acceleration |
 | of the node for the given index.                         hahn 09/13 |
 *---------------------------------------------------------------------*/
double DRT::UTILS::AccelerationProfileFunction::Evaluate(int index, const double* xp, double t, DRT::Discretization* dis)
{
    // Determine time difference
    double deltaT = t - timeOld_;

    // If new time step, determine current acceleration
    // *****************************************************************************
    if (deltaT > 1e-9) { // new time step

        // Determine current acceleration (at t)
        // -------------------------------------------------------------------------
        acc_B_(0,0) = 0.0;
        acc_B_(1,0) = 0.0;
        acc_B_(2,0) = 0.0;

        for (int i=0; i<numAccelerations_; i++) {
            if (t >= accelerations_[i*NUMACCELERATIONCELLS_+0]) {
                acc_B_(0,0) = accelerations_[i*NUMACCELERATIONCELLS_+1];
                acc_B_(1,0) = accelerations_[i*NUMACCELERATIONCELLS_+2];
                acc_B_(2,0) = accelerations_[i*NUMACCELERATIONCELLS_+3];
            }
        }

        // Update time of last time step
        // -------------------------------------------------------------------------
        timeOld_ = t;
    }

    // Return the acceleration of the node for the given index
    // *****************************************************************************
    return acc_B_(index,0);
}
