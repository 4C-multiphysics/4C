#ifdef D_SHELL8
#include "../headers/standardtypes.h"
#include "shell8.h"
/*----------------------------------------------------------------------*
 | st.venant-kirchhoff-material                           m.gee 6/01    |
 *----------------------------------------------------------------------*/
void s8_mat_linel(STVENANT *mat, double **g, double **CC)
{
int i,j,k,l;
double xsi=1.0; /*----- shear correction coefficient not yet introduced */
double C[3][3][3][3]; /*--------------------------- constitutive tensor */
double l1,l2;/*----------------------------------------- lame constants */
double emod,nue;/*--------------------------------------- mat constants */
#ifdef DEBUG 
dstrc_enter("s8_mat_linel");
#endif
/*----------------------------------------------------------------------*/
emod = mat->youngs;
nue  = mat->possionratio;
l1 = (emod*nue) / ((1.0+nue)*(1.0-2.0*nue));
l2 = emod/ (2.0*(1.0+nue));
/*---------this is not very fast, but corresponds nicely with theory... */
for (i=0; i<3; i++)
for (j=0; j<3; j++)
for (k=0; k<3; k++)
for (l=0; l<3; l++)
C[i][j][k][l] = l1*g[i][j]*g[k][l] + l2*( g[i][k]*g[j][l]+g[i][l]*g[k][j] );
/*----------------------------------------------------------------------*/
CC[0][0] = C[0][0][0][0];
CC[0][1] = C[0][0][1][0];
CC[0][2] = C[0][0][2][0];
CC[0][3] = C[0][0][1][1];
CC[0][4] = C[0][0][2][1];
CC[0][5] = C[0][0][2][2];

CC[1][0] = C[1][0][0][0];
CC[1][1] = C[1][0][1][0];
CC[1][2] = C[1][0][2][0];
CC[1][3] = C[1][0][1][1];
CC[1][4] = C[1][0][2][1];
CC[1][5] = C[1][0][2][2];

CC[2][0] = C[2][0][0][0];
CC[2][1] = C[2][0][1][0];
CC[2][2] = C[2][0][2][0]/*/xsi*/;
CC[2][3] = C[2][0][1][1];
CC[2][4] = C[2][0][2][1]/*/xsi*/;
CC[2][5] = C[2][0][2][2];

CC[3][0] = C[1][1][0][0];
CC[3][1] = C[1][1][1][0];
CC[3][2] = C[1][1][2][0];
CC[3][3] = C[1][1][1][1];
CC[3][4] = C[1][1][2][1];
CC[3][5] = C[1][1][2][2];

CC[4][0] = C[2][1][0][0];
CC[4][1] = C[2][1][1][0];
CC[4][2] = C[2][1][2][0]/*/xsi*/;
CC[4][3] = C[2][1][1][1];
CC[4][4] = C[2][1][2][1]/*/xsi*/;
CC[4][5] = C[2][1][2][2];

CC[5][0] = C[2][2][0][0];
CC[5][1] = C[2][2][1][0];
CC[5][2] = C[2][2][2][0];
CC[5][3] = C[2][2][1][1];
CC[5][4] = C[2][2][2][1];
CC[5][5] = C[2][2][2][2];
/*----------------------------------------------------------------------*/
#ifdef DEBUG 
dstrc_exit();
#endif
return;
} /* end of s8_mat_linel */
/*----------------------------------------------------------------------*
 | PK II stresses                                         m.gee 6/01    |
 *----------------------------------------------------------------------*/
void s8_mat_stress1(double *stress, double *strain, double **C)
{
double E[6];
#ifdef DEBUG 
dstrc_enter("s8_mat_linel");
#endif
/*----------------------------------------------------------------------*/
E[0] = strain[0];
E[3] = strain[3];
E[5] = strain[5];
E[1] = strain[1] * 2.0;
E[2] = strain[2] * 2.0;
E[4] = strain[4] * 2.0;
math_matvecdense(stress,C,E,6,6,0,1.0);
/*----------------------------------------------------------------------*/
#ifdef DEBUG 
dstrc_exit();
#endif
return;
} /* end of s8_mat_linel */
/*----------------------------------------------------------------------*
 | neohooke material from habil wriggers                  m.gee 3/03    |
 *----------------------------------------------------------------------*/
void s8_mat_neohooke(NEO_HOOKE *mat, 
                     double    *stress, 
                     double   **CC,
                     double   **gmkonr,
                     double   **gmkonc,
                     double     detr,
                     double     detc)
{
int i,j,k,l;
double xsi=1.0; /*----- shear correction coefficient not yet introduced */
double C[3][3][3][3]; /*--------------------------- constitutive tensor */
double sp[3][3];/*---------------------------------------- PK2 stresses */
double l1,l2;/*----------------------------------------- lame constants */
double emod,nue;/*--------------------------------------- mat constants */
double xj;
double F1;
#ifdef DEBUG 
dstrc_enter("s8_mat_neohooke"); 
#endif
/*----------------------------------------------------------------------*/
emod = mat->youngs;
nue  = mat->possionratio;
l1 = (emod*nue) / ((1.0+nue)*(1.0-2.0*nue));
l2 = emod/ (2.0*(1.0+nue));
xj = detc/detr;
if (xj < EPS6) xj = EPS6;
F1 = l1 * log(xj) - l2;
/*-------------------------------------------------------- pk2 stresses */
for (i=0; i<3; i++)
for (j=0; j<3; j++)
   sp[i][j] = F1 * gmkonc[i][j] + l2 * gmkonr[i][j];
stress[0] = sp[0][0];   
stress[1] = sp[0][1];   
stress[2] = sp[0][2];   
stress[3] = sp[1][1];   
stress[4] = sp[1][2];   
stress[5] = sp[2][2];   
/*---------this is not very fast, but corresponds nicely with theory... */
for (i=0; i<3; i++)
for (j=0; j<3; j++)
for (k=0; k<3; k++)
for (l=0; l<3; l++)
C[i][j][k][l] = l1*gmkonc[i][j]*gmkonc[k][l] - F1*(gmkonc[i][k]*gmkonc[j][l]+gmkonc[i][l]*gmkonc[k][j]);
/*----------------------------------------------------------------------*/
CC[0][0] = C[0][0][0][0];         
CC[0][1] = C[0][0][1][0];
CC[0][2] = C[0][0][2][0];
CC[0][3] = C[0][0][1][1];
CC[0][4] = C[0][0][2][1];
CC[0][5] = C[0][0][2][2];

CC[1][0] = C[1][0][0][0];
CC[1][1] = C[1][0][1][0];
CC[1][2] = C[1][0][2][0];
CC[1][3] = C[1][0][1][1];
CC[1][4] = C[1][0][2][1];
CC[1][5] = C[1][0][2][2];

CC[2][0] = C[2][0][0][0];
CC[2][1] = C[2][0][1][0];
CC[2][2] = C[2][0][2][0]/*/xsi*/;
CC[2][3] = C[2][0][1][1];
CC[2][4] = C[2][0][2][1]/*/xsi*/;
CC[2][5] = C[2][0][2][2];

CC[3][0] = C[1][1][0][0];
CC[3][1] = C[1][1][1][0];
CC[3][2] = C[1][1][2][0];
CC[3][3] = C[1][1][1][1];
CC[3][4] = C[1][1][2][1];
CC[3][5] = C[1][1][2][2];

CC[4][0] = C[2][1][0][0];
CC[4][1] = C[2][1][1][0];
CC[4][2] = C[2][1][2][0]/*/xsi*/;
CC[4][3] = C[2][1][1][1];
CC[4][4] = C[2][1][2][1]/*/xsi*/;
CC[4][5] = C[2][1][2][2];

CC[5][0] = C[2][2][0][0];
CC[5][1] = C[2][2][1][0];
CC[5][2] = C[2][2][2][0];
CC[5][3] = C[2][2][1][1];
CC[5][4] = C[2][2][2][1];
CC[5][5] = C[2][2][2][2];
/*----------------------------------------------------------------------*/
#ifdef DEBUG 
dstrc_exit();
#endif
return;
} /* end of s8_mat_neohooke */



#endif
