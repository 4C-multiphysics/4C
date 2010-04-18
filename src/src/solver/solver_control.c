/*!----------------------------------------------------------------------
\file
\brief

<pre>
Maintainer: Malte Neumann
            neumann@statik.uni-stuttgart.de
            http://www.uni-stuttgart.de/ibs/members/neumann/
            0711 - 685-6121
</pre>

*----------------------------------------------------------------------*/
#ifndef CCADISCRET
#include "../headers/standardtypes.h"
#include "../solver/solver.h"

/*!----------------------------------------------------------------------
\brief file pointers

<pre>                                                         m.gee 8/00
This structure struct _FILES allfiles is defined in input_control_global.c
and the type is in standardtypes.h
It holds all file pointers and some variables needed for the FRSYSTEM
</pre>
*----------------------------------------------------------------------*/
extern struct _FILES  allfiles;
/*----------------------------------------------------------------------*
 |                                                       m.gee 06/01    |
 | general problem data                                                 |
 | struct _GENPROB       genprob; defined in global_control.c           |
 *----------------------------------------------------------------------*/
extern struct _GENPROB     genprob;
/*----------------------------------------------------------------------*
 | define the global structure solv                                     |
 |                                                                      |
 | global variable *solv, vector of lenght numfld of structures SOLVAR  |
 |                                                       m.gee 11/00    |
 *----------------------------------------------------------------------*/
 struct _SOLVAR  *solv;


#ifdef TRILINOS_PACKAGE
/* Trilinos solver interface defined in solver_trilinos_control.cpp */
extern void solver_trilinos_control(struct _FIELD          *actfield,
                                    int                     disnum,
                                    struct _SOLVAR         *actsolv,
                                    struct _INTRA          *actintra,
                                    enum   _SPARSE_TYP     *sysarray_typ,
                                    union  _SPARSE_ARRAY   *sysarray,
                                    struct _DIST_VECTOR    *sol,
                                    struct _DIST_VECTOR    *rhs,
                                    INT                     option);
#endif

/*----------------------------------------------------------------------*
 |  routine to control all solver calls                  m.gee 9/01     |
 *----------------------------------------------------------------------*/
void solver_control(
                       struct _FIELD          *actfield,
                       INT                     disnum,
                       struct _SOLVAR         *actsolv,
                       struct _INTRA          *actintra,
                       enum   _SPARSE_TYP     *sysarray_typ,
                       union  _SPARSE_ARRAY   *sysarray,
                       struct _DIST_VECTOR    *sol,
                       struct _DIST_VECTOR    *rhs,
                       INT                     option
                      )
{
DOUBLE t0,t1;
#ifdef DEBUG
dstrc_enter("solver_control");
#endif

#ifdef PERF
  perf_begin(20);
#endif

/*----------------------------------------------------------------------*/
#ifndef NO_PRINT_SOLVER_TIME
t0=ds_cputime();
#endif
/*----------------------------------------------------------------------*/
if (genprob.usetrilinosalgebra) /* Use Trilinos solver interfaces */
{
#ifdef TRILINOS_PACKAGE
  solver_trilinos_control(actfield,disnum,actsolv,actintra,sysarray_typ,sysarray,sol,rhs,option);
#endif
}
else /* Use ccarat generic interfaces */
{
  switch(*sysarray_typ)
  {

  #ifdef MLIB_PACKAGE
  case mds:/*-------------------------------- system matrix is mds matrix */
     solver_mlib(actsolv,actintra,sysarray->mds,sol,rhs,option);
  break;
  #endif

  #ifdef AZTEC_PACKAGE
  case msr:/*-------------------------------- system matrix is msr matrix */
   solver_az_msr(actsolv,actintra,sysarray->msr,sol,rhs,option);
  break;
  #endif

  #ifdef HYPRE_PACKAGE
  case parcsr:/*-------------------------- system matrix is parcsr matrix */
     solver_hypre_parcsr(actsolv,actintra,sysarray->parcsr,sol,rhs,option);
  break;
  #endif

  #ifdef PARSUPERLU_PACKAGE
  case ucchb:/*---------------------------- system matrix is ucchb matrix */
     solver_psuperlu_ucchb(actsolv,actintra,sysarray->ucchb,sol,rhs,option);
  break;
  #endif

  case dense:/*---------------------------- system matrix is dense matrix */
     solver_lapack_dense(actsolv,actintra,sysarray->dense,sol,rhs,option);
  break;

  #ifdef MUMPS_PACKAGE
  case rc_ptr:/*---------------------- system matrix is row/column matrix */
     solver_mumps(actsolv,actintra,sysarray->rc_ptr,sol,rhs,option);
  break;
  #endif

  #ifdef UMFPACK
  case ccf:/*------------------ system matrix is compressed column matrix */
     solver_umfpack(actsolv,actintra,sysarray->ccf,sol,rhs,option);
  break;
  #endif

  case skymatrix:/*---------------------- system matrix is skyline matrix */
     solver_colsol(actsolv,actintra,sysarray->sky,sol,rhs,option);
  break;

  #ifdef SPOOLES_PACKAGE
  case spoolmatrix:/*-------------------- system matrix is spooles matrix */
     solver_spooles(actsolv,actintra,sysarray->spo,sol,rhs,option);
  break;
  #endif

  #ifdef MLPCG
  case bdcsr:/*------------------------------ system matrix is csr matrix */
     solver_mlpcg(actsolv,actintra,sysarray->bdcsr,sol,rhs,option);
  break;
  #endif

  case oll:/*------------------------------ system matrix is oll matrix */
     solver_oll(actsolv,actintra,sysarray->oll,sol,rhs,option);
  break;

  default:
     dserror("Unknown format typ of system matrix");
  break;
  }
}
/*----------------------------------------------------------------------*/
#ifndef NO_PRINT_SOLVER_TIME
t1=ds_cputime();
t1 -= t0;
fprintf(allfiles.out_err,"Time for this solver call: %f\n",t1);
fflush(allfiles.out_err);
#endif
/*----------------------------------------------------------------------*/

#ifdef PERF
  perf_end(20);
#endif

#ifdef DEBUG
dstrc_exit();
#endif

return;
} /* end of solver_control */

#endif  /* #ifndef CCADISCRET */
