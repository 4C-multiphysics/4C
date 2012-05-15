/*!---------------------------------------------------------------------
\file
\brief

<pre>
Maintainer: Malte Neumann
            neumann@statik.uni-stuttgart.de
            http://www.uni-stuttgart.de/ibs/members/neumann/
            0711 - 685-6121
</pre>

---------------------------------------------------------------------*/
#if defined(D_SHELL8)
typedef enum _MATERIAL_TYP
{
                       m_stvenant,    /* St.Venant Kirchhoff material */
                       m_thermostvenant,    /* St.Venant Kirchhoff material with temperature */
                       m_pl_mises_3D, /* Stefans Mises*/
                       m_pl_mises,    /* von Mises material */
                       m_pl_hoff,     /* anisotropic plastic material based on hoffman criterion */
                       m_damage,      /* 3D damage matieral */
                       m_pl_foam,     /* foam material - large strains */
                       m_pl_mises_ls, /* von Mises material - large strains*/
                       m_pl_dp,       /* Drucker Prager material */
                       m_pl_epc,      /* elastoplastic concrete material */
                       m_pl_epc3D,    /* elastoplastic concrete material 3D formulation */
                       m_stvenpor,    /* porous St.Venant Kirchhoff material */
                       m_pl_por_mises,/* porous von Mises material */
                       m_neohooke,    /* Neo-Hooke material */
                       m_aaaneohooke, /* quasi Neo-Hooke material for aneurysmatic artery wall */
                       m_aaaraghavanvorp_damage, /* quasi Neo-Hooke material for aneurysmatic artery wall with damage*/
                       m_compogden,   /* compressible Ogden material (with shell8) */
                       m_viscohyper,  /* compressible viscous Ogden material (with shell8) */
                       m_fluid,       /* fluid */
                       m_sutherland_fluid,  /* fluid material with temperature dependence according to Sutherland law */
                       m_carreauyasuda,/* fluid with nonlinear viscosity according to Carreau-Yasuda*/
                       m_modpowerlaw,  /* fluid with nonlinear viscosity according to a modified power law*/
                       m_condif,      /* convection-diffusion */
                       m_sutherland_condif,  /* convection-diffusion material with temperature dependence according to Sutherland law */
                       m_pl_hash,     /* elpl. hashin delamination material */
                       m_el_orth,     /* elastic orthotropic material */
                       m_mfoc,        /* open cell metal foam */
                       m_mfcc,        /* closed cell metal foam */
                       m_nhmfcc,      /* foam, closed cell, based on modified Neo Hook */
                       m_multi_layer, /* multilayer material -> shell9*/
                       m_ifmat,        /* interface surface elasto-damage-plasto material*/
                       m_interf_therm, /* themodyn. based interface elasto-damage surface material*/
                       m_dam_mp,       /* isotropic damage model -> mazars/pijadier-cabot*/
                       m_damage_ge,    /* isotropic gradient enhanced damage model */
                       m_itskov,       /* hyperelastic polyconvex energy strain function following Itskov */
                       m_anisotropic_balzani,  /* anisotropic polyconvex material*/
                       m_mooneyrivlin,  /* Mooney-Rivlin material*/
                       m_yeoh,          /* Yeoh material*/
                       m_elasthyper, /* Hyperelastic toolbox */
                       m_viscogenmax, /* Viscous Generalized Maxwell model compatible with hyperelastic toolbox */
                       m_visconeohooke, /* Viscous NeoHookean Material */
                       m_viscoanisotropic, /* Viscous Anisotropic Fiber Material */
                       m_contchainnetw, /* Continuum Chain Network Material Law with remodeling */
                       m_artwallremod,  /* Arterial Wall Material Law (Holzapfel) with remodeling (Hariton) */
                       m_th_fourier_iso,  /* isotropic (linear) Fourier's law of heat conduction */
                       m_th_fourier_gen,  /* general (linear) Fourier's law of heat conduction */
                       m_vp_robinson,   /* Robinson's visco-plastic material */
                       m_struct_multiscale, /*  structural microscale approach */
                       m_matlist,       /* collection of single materials (used for scalar transport problems)*/
                       m_biocell,       /* biological cell model */
                       m_ion,           /* properties of an ion species in an electrolyte solution */
                       m_cnst_art,      /* 1D_Artery with constant material and geometric properties */
                       m_holzapfelcardiovascular, /* anisotropic fiber material for arteries */
                       m_humphreycardiovascular /* anisotropic material for arteries cf Humphrey */
} MATERIAL_TYP;

/*----------------------------------------------------------------------*
 | materials                                              m.gee 4/01    |
 | structure to hold all types of material laws                         |
 *----------------------------------------------------------------------*/
typedef struct _MATERIAL
{
     INT                       Id;           /* Id of the material */

     enum _MATERIAL_TYP        mattyp;       /* type of material */

     union
     {
     struct _STVENANT            *stvenant;     /* St. Venant-Kirchhoff material */
     struct _PL_HOFF             *pl_hoff;      /* anisotropic plastic material, based on hoffman-criterion */
     struct _PL_MISES            *pl_mises;     /* von Mises material */
     struct _DAMAGE              *damage;       /* CDM material */
     struct _PL_FOAM             *pl_foam;      /* foam material - large strains */
     struct _PL_MISES_LS         *pl_mises_ls;  /* von Mises material - large strains*/
     struct _PL_DP               *pl_dp;        /* Drucker Prager material */
     struct _PL_EPC              *pl_epc;       /* elastoplastic concrete material */
     struct _STVENPOR            *stvenpor;     /* porous St. Ven.-Kirch. material */
     struct _PL_POR_MISES        *pl_por_mises; /* porous von Mises material */
     struct _NEO_HOOKE           *neohooke;     /* Neo-Hooke material */
     struct _AAA_NEO_HOOKE       *aaaneohooke;  /* quasi Neo-Hooke material for aneurysmatic artery wall*/
     struct _COMPOGDEN           *compogden;    /* compressible ogden hyperelastic material */
     struct _VISCOHYPER          *viscohyper;   /* viscoelastic compressible ogden hyperelastic material */
     struct _FLUID               *fluid;        /* fluid material */
     struct _SUTHERLAND_FLUID    *sutherland_fluid;  /* fluid material with temperature dependence according to Sutherland law */
     struct _CARREAUYASUDA       *carreauyasuda;/* fluid with nonlinear viscosity according to Carreau-Yasuda */
     struct _MODPOWERLAW         *modpowerlaw;  /* fluid with nonlinear viscosity according to modified power law */
     struct _CONDIF              *condif;       /* convection-diffusion material */
     struct _SUTHERLAND_CONDIF   *sutherland_condif;  /* convection-diffusion material with temperature dependence according to Sutherland law */
     struct _PL_HASH             *pl_hash;      /* elpl. hashin delamination material */
     struct _EL_ORTH             *el_orth;      /* elastic orthotropic material */
     struct _MFOC                *mfoc;         /* metal foam, open cell  */
     struct _MFCC                *mfcc;         /* metal foam, closed cell  */
     struct _NHMFCC              *nhmfcc;       /* foam, closed cell, based on modified Neo Hook */
     struct _MULTI_LAYER         *multi_layer;  /* multi layer material*/
     struct _IFMAT               *ifmat;        /* interface elasto-damage-plasto surface material*/
     struct _INTERF_THERM        *interf_therm; /* themodyn. based interface elasto-damage surface material*/
     struct _DAM_MP              *dam_mp;       /* isotropic damage material (mazars-pijadier-cabot)*/
     struct _DAMAGE_GE           *damage_ge;    /* isotropic gradient enhanced damage material */
     struct _ITSKOV              *itskov;       /* Itskov material for isotropic case */
     struct _ANISOTROPIC_BALZANI *anisotropic_balzani; /* anisotropic hyperelastic polyconvex material */
     struct _MOONEYRIVLIN        *mooneyrivlin; /* Mooney-Rivlin material */
     struct _YEOH                *yeoh; /* Mooney-Rivlin material */
     struct _ELASTHYPER          *elasthyper; /* Hyperelastic toolbox */
     struct _VISCOGENMAX         *viscogenmax; /* Viscous Generalized Maxwell model compatible with hyperelastic toolbox */
     struct _VISCONEOHOOKE       *visconeohooke; /* Viscous NeoHookean material */
     struct _VISCOANISOTROPIC    *viscoanisotropic; /* Viscous Anisotropic fiber material */
     struct _CONTCHAINNETW       *contchainnetw; /* Continuum Chain Network material law */
     struct _ARTWALLREMOD        *artwallremod;  /* Arterial Wall Material Law with Remodeling */
     struct _TH_FOURIER_ISO      *th_fourier_iso;   /* isotropic Fourier's law of heat conduction */
     struct _TH_FOURIER_GEN      *th_fourier_gen;   /* general heat conduction matrix of Fourier's (linear) law of heat conduction */
     struct _VP_ROBINSON         *vp_robinson;   /* viscoplastic Robinson material */
     struct _STRUCT_MULTISCALE   *struct_multiscale;     /* material parameters are calculated from microscale simulation */
     struct _MATLIST             *matlist;       /* collection of any single materials (used for scalar transport problems)*/
     struct _BIOCELL             *biocell;       /* Biological Cell model */
     struct _ION                 *ion;           /* material properties of an ion species in an electrolyte solution */
     struct _CNST_ART            *cnst_art;      /* 1D_Artery with constant material and geometric properties */
     struct _HOLZAPFELCARDIO     *holzapfelcardiovascular; /* anisotropic fiber material for arteries */
     struct _HUMPHREYCARDIO      *humphreycardiovascular;  /* anisotropic material for arteries cf Humphrey */
     }                           m;             /* union pointer to material specific structure */

} MATERIAL;

/*----------------------------------------------------------------------*
 | interpolation types of non-const parameters              bborn 03/07 |
 *----------------------------------------------------------------------*/
typedef enum _MAT_PARAM_INTPOL
{
     mat_param_ipl_none,                  /* no interpolation */
     mat_param_ipl_const,                 /* constant, allright, this
                                           * is polynomial as well,
                                           * but can be treated quickly */
     mat_param_ipl_poly,                  /* polynomial */
     mat_param_ipl_pcwslnr,               /* piecewise linear */
     mat_param_ipl_exp                    /* exponential */
} MAT_PARAM_INTPOL;

/*----------------------------------------------------------------------*
 | data type of non-const multiple parameters               bborn 04/07 |
 *----------------------------------------------------------------------*/
typedef struct _MAT_PARAM_MULT
{
     MAT_PARAM_INTPOL          ipl;            /* interpolation type */
     INT                       n;              /* number of parameter
                                                * data components */
     DOUBLE*                   d;              /* data array */
} MAT_PARAM_MULT;


/*----------------------------------------------------------------------*
 | St. Venant-Kirchhoff material                          m.gee 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _STVENANT
{
     DOUBLE                    youngs;         /* Young's modulus */
     DOUBLE                    possionratio;   /* Possion ratio */
     DOUBLE                    density;        /* material specific weight */
     DOUBLE                    thermexpans;    /* coefficient of thermal expansion */
} STVENANT;


/*----------------------------------------------------------------------*
 | porous St. Venant-Kirchhoff material (optimization)       al 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _STVENPOR
{
     DOUBLE                    youngs;         /* Young's modulus */
     DOUBLE                    possionratio;   /* Possion ratio */
     DOUBLE                    density;        /* material specific weight */
     DOUBLE                    refdens;        /* reference density */
     DOUBLE                    exponent;       /* material parameter */
} STVENPOR;


/*----------------------------------------------------------------------*
 | Neo Hooke material                                     m.gee 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _NEO_HOOKE
{
     DOUBLE                    youngs;         /* Young's modulus */
     DOUBLE                    possionratio;   /* Possion ratio */
     DOUBLE                    density;        /* material specific weight */
} NEO_HOOKE;

/*----------------------------------------------------------------------*
 | AAA generalised Neo Hooke material                     chfoe 4/08    |
 *----------------------------------------------------------------------*/
typedef struct _AAA_NEO_HOOKE
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    beta;          /* 2nd parameter */
     DOUBLE                    nue;           /* Poisson's ratio */
     DOUBLE                    density;       /* material specific weight */
} AAA_NEO_HOOKE;

/*----------------------------------------------------------------------*
 | AAA raghavan & vorp + ogden material with damage       chfoe 4/08    |
 *----------------------------------------------------------------------*/
typedef struct _AAA_RAG_DAMAGE
{
     DOUBLE 			bulk; 	      /* Bulk's modulus (Volumetric)*/
     DOUBLE 			alpha;        /* 1st parameter, alpha (Isochoric)*/
     DOUBLE 			beta;         /* 2nd parameter, beta (Isochoric)*/
     DOUBLE 			eqstrmin;     /* equivalent strain initial damage*/
     DOUBLE 			a;            /* 1st parameter, a*/
     DOUBLE 			b;            /* 2nd parameter, b*/
     DOUBLE 			density;      /* mass density*/
} AAA_RAG_DAMAGE;

/*----------------------------------------------------------------------*
 | compressible ogden material                            m.gee 6/03    |
 *----------------------------------------------------------------------*/
typedef struct _COMPOGDEN
{
     INT                       init;           /* init flag */
     DOUBLE                    nue;            /* Possion ratio */
     DOUBLE                    beta;           /* the unphysical material constant called beta */
     DOUBLE                    alfap[3];       /* three parameters alfap */
     DOUBLE                    mup[3];         /* three parameters nuep */
     DOUBLE                    density;        /* material specific weight */
     DOUBLE                    lambda;         /* 1. lame constant */
     DOUBLE                    kappa;          /* bulkmodulus */
#if 1
     DOUBLE                    l[3];
#endif
} COMPOGDEN;

/*----------------------------------------------------------------------*
 | viscoelastic compressible ogden material               m.gee 9/03    |
 *----------------------------------------------------------------------*/
typedef struct _VISCOHYPER
{
     INT                       init;           /* init flag */
     DOUBLE                    nue;            /* Possion ratio */
     DOUBLE                    beta;           /* the unphysical material constant called beta */
     DOUBLE                    alfap[3];       /* three parameters alfap */
     DOUBLE                    mup[3];         /* three parameters nuep */
     DOUBLE                    density;        /* material specific weight */
     DOUBLE                    lambda;         /* 1. lame constant */
     DOUBLE                    kappa;          /* bulkmodulus */
     INT                       nmaxw;          /* number of maxwell elements in the material (1-4) */
     DOUBLE                    tau[4];         /* relaxation times of hte maxwell elements */
     DOUBLE                    betas[4];       /* strain energy factors of the springs of the maxwell elements */
} VISCOHYPER;

/*----------------------------------------------------------------------*
 | fluid material                                         m.gee 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _FLUID
{
     DOUBLE                    viscosity; /* kinematic or dynamic viscosity */
     DOUBLE                    density;
     DOUBLE                    gamma;     /* surface tension coeficient */
} FLUID;


/*----------------------------------------------------------------------*
 | fluid material according to Sutherland law               vg 01/09    |
 *----------------------------------------------------------------------*/
typedef struct _SUTHERLAND_FLUID
{
     DOUBLE                    refvisc;  /* reference dynamic viscosity (kg/(m*s)) */
     DOUBLE                    reftemp;  /* reference temperature (K) */
     DOUBLE                    suthtemp; /* Sutherland temperature (K) */
} SUTHERLAND_FLUID;


/*----------------------------------------------------------------------*
 | fluid with nonlinear viscosity according               u.may 4/08    |
 | to Carreau-Yasuda                                                    |
 *----------------------------------------------------------------------*/
typedef struct _CARREAUYASUDA
{
     DOUBLE                    nu_0; 		/* zero-shear viscosity */
     DOUBLE                    nu_inf;		/* infinite-shear viscosity */
     DOUBLE                    lambda;      /* characteristic time */
     DOUBLE                    a_param;		/* constant parameter */
     DOUBLE                    b_param;     /* constant parameter */
     DOUBLE                    density;     /* density */
} CARREAUYASUDA;


/*----------------------------------------------------------------------*
 | fluid with nonlinear viscosity according               u.may 4/08    |
 | to a modified power law                                              |
 *----------------------------------------------------------------------*/
typedef struct _MODPOWERLAW
{
     DOUBLE                    m_cons; 		    /* consistency */
     DOUBLE                    delta;       	/* safety factor */
     DOUBLE                    a_exp;			/* exponent */
     DOUBLE                    density;     	/* density */
} MODPOWERLAW;


/*----------------------------------------------------------------------*
 | convection-diffusion material                             vg 6/07    |
 *----------------------------------------------------------------------*/
typedef struct _CONDIF
{
     DOUBLE                    diffusivity; /* kinematic diffusivity */
     DOUBLE                    shc;     /* specific heat capacity */
} CONDIF;

/*----------------------------------------------------------------------*
 | convection-diffusion material according to Sutherland law  vg 01/09  |
 *----------------------------------------------------------------------*/
typedef struct _SUTHERLAND_CONDIF
{
     DOUBLE                    refvisc;  /* reference dynamic viscosity (kg/(m*s)) */
     DOUBLE                    reftemp;  /* reference temperature (K) */
     DOUBLE                    suthtemp; /* Sutherland temperature (K) */
     DOUBLE                    shc;      /* specific heat capacity */
     DOUBLE                    pranum;   /* Prandtl number */
} SUTHERLAND_CONDIF;


/*----------------------------------------------------------------------*
 | plastic mises material                              a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_MISES
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    possionratio;  /* Possion ratio */
     DOUBLE                    ALFAT;
     DOUBLE                    Sigy;
     DOUBLE                    Hard;
     DOUBLE                    GF;
     DOUBLE                    betah;
} PL_MISES;

/*----------------------------------------------------------------------*
 | Damage material                                     he      04/03    |
 *----------------------------------------------------------------------*/
typedef struct _DAMAGE
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    possionratio;  /* Possion ratio */
     INT                       Equival;
     INT                       Damtyp;
     DOUBLE                    Kappa_0;
     DOUBLE                    Kappa_m;
     DOUBLE                    Alpha;
     DOUBLE                    Beta;
     DOUBLE                    k_fac;
} DAMAGE;
/*----------------------------------------------------------------------*
 | anisotropic plastic material based on hoffman-criterion    sh 03/03  |
 *----------------------------------------------------------------------*/
typedef struct _PL_HOFF
{
     DOUBLE                    emod1;
     DOUBLE                    emod2;
     DOUBLE                    emod3;
     DOUBLE                    gmod12;
     DOUBLE                    gmod13;
     DOUBLE                    gmod23;
     DOUBLE                    xnue12;
     DOUBLE                    xnue13;
     DOUBLE                    xnue23;
     DOUBLE                    s11T;
     DOUBLE                    s11C;
     DOUBLE                    s22T;
     DOUBLE                    s22C;
     DOUBLE                    s33T;
     DOUBLE                    s33C;
     DOUBLE                    s12;
     DOUBLE                    s23;
     DOUBLE                    s13;
     DOUBLE                    uniax;
     DOUBLE                    sh11T;
     DOUBLE                    sh11C;
     DOUBLE                    sh22T;
     DOUBLE                    sh22C;
     DOUBLE                    sh33T;
     DOUBLE                    sh33C;
     DOUBLE                    sh12;
     DOUBLE                    sh23;
     DOUBLE                    sh13;
     DOUBLE                    ha11T;
     DOUBLE                    ha11C;
     DOUBLE                    ha22T;
     DOUBLE                    ha22C;
     DOUBLE                    ha33T;
     DOUBLE                    ha33C;
     DOUBLE                    ha12;
     DOUBLE                    ha23;
     DOUBLE                    ha13;
} PL_HOFF;

/*----------------------------------------------------------------------*
 | plastic mises material including large strains      a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_MISES_LS
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    possionratio;  /* Possion ratio */
     DOUBLE                    ALFAT;
     DOUBLE                    Sigy;
     DOUBLE                    Hard;
     DOUBLE                    GF;
} PL_MISES_LS;

/*----------------------------------------------------------------------*
 | plastic foam  material                              a.lipka 02/12    |
 *----------------------------------------------------------------------*/
typedef struct _PL_FOAM
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    possionratio;  /* Possion ratio */
     DOUBLE                    ALFAT;
     DOUBLE                    Sigy;
     DOUBLE                    Hard;
     DOUBLE                    GF;
} PL_FOAM;
/*----------------------------------------------------------------------*
 | plastic drucker prager material                     a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_DP
{
     DOUBLE                    youngs;        /* Young's modulus */
     DOUBLE                    possionratio;  /* Possion ratio */
     DOUBLE                    ALFAT;
     DOUBLE                    Sigy;
     DOUBLE                    Hard;
     DOUBLE                    PHI;
     DOUBLE                    GF;
     DOUBLE                    betah;
} PL_DP;
/*----------------------------------------------------------------------*
 | elastoplastic concrete material                     a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_EPC
{
     DOUBLE                    dens;
     /* concrete */
     DOUBLE                    youngs;       /* Young's modulus */
     DOUBLE                    possionratio; /* Possion ratio */
     DOUBLE                    alfat;
     DOUBLE                    sigy;
     DOUBLE                    phi;
     DOUBLE                    xsi;
     DOUBLE                    ftm;        /* tensile strength */
     DOUBLE                    gt;         /* tensile fracture energy */
     DOUBLE                    fcm;        /* compressive strength */
     DOUBLE                    gc;         /* compressive fracture energy */
     DOUBLE                    gamma1;     /* fitting factor yield function 1 */
     DOUBLE                    gamma2;     /* symm. biaxial compression stressfactor */
     DOUBLE                    gamma3;     /* fitting parameter to account for HPC -> Haufe (=1/3 for normal concrete)*/
     DOUBLE                    gamma4;     /* fitting parameter to account for HPC -> Haufe (=4/3 for normal concrete*/
     DOUBLE                    dfac;       /* dammage factor: 0.0 plastic - 1.0 full damaged */
     /* tension stiffening */
     INT                       nstiff;     /* ==1 in consideration of tension stiffening */
     /* rebars */
     INT                       maxreb;     /* number of*/
     INT                      *rebar;      /* Id */
     DOUBLE                   *reb_area;   /* area   */
     DOUBLE                   *reb_ang;    /* angel  */
     DOUBLE                   *reb_so;     /* minimum bond length  */
     DOUBLE                   *reb_ds;     /* diameter  */
     DOUBLE                   *reb_rgamma; /* =4: deformed bars =2: plane par */
     DOUBLE                   *reb_dens;
     DOUBLE                   *reb_alfat;
     DOUBLE                   *reb_emod;
     DOUBLE                   *reb_rebnue;
     DOUBLE                   *reb_sigy;
     DOUBLE                   *reb_hard;
} PL_EPC;
/*----------------------------------------------------------------------*
 | plastic mises porous material                       a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_POR_MISES
{
     DOUBLE                    youngs;
     DOUBLE                    DP_YM;
     DOUBLE                    possionratio;
     DOUBLE                    ALFAT;
     DOUBLE                    Sigy;
     DOUBLE                    DP_Sigy;
     DOUBLE                    Hard;
     DOUBLE                    DP_Hard;
} PL_POR_MISES;
/*----------------------------------------------------------------------*
 | delamination material                               a.lipka 17/05    |
 *----------------------------------------------------------------------*/
typedef struct _PL_HASH
{
     DOUBLE                    emod1;
     DOUBLE                    emod2;
     DOUBLE                    emod3;
     DOUBLE                    xnue23;
     DOUBLE                    xnue13;
     DOUBLE                    xnue12;
     DOUBLE                    gmod12;
     DOUBLE                    gmod23;
     DOUBLE                    gmod13;
     DOUBLE                    s33;
     DOUBLE                    sinf33;
     DOUBLE                    s23;
     DOUBLE                    s13;
     DOUBLE                    gamma;
     DOUBLE                    gc;
     DOUBLE                    deltat;
     DOUBLE                    eta_i;
     DOUBLE                    c1hgt;
     DOUBLE                    c1layhgt;
     INT                       ivisco;
} PL_HASH;

/*----------------------------------------------------------------------*
 | elastic orthotropic material                              al 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _EL_ORTH
{
     DOUBLE                    emod1;
     DOUBLE                    emod2;
     DOUBLE                    emod3;
     DOUBLE                    xnue23;
     DOUBLE                    xnue13;
     DOUBLE                    xnue12;
     DOUBLE                    gmod12;
     DOUBLE                    gmod23;
     DOUBLE                    gmod13;
} EL_ORTH;

/*----------------------------------------------------------------------*
 | open cell metal foam material (optimization)              al 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _MFOC
{
     DOUBLE                    es;             /* Young's modulus (cell) */
     DOUBLE                    pr;             /* Possion ratio */
     DOUBLE                    dens;           /* density foam  */
     DOUBLE                    denss;          /* density (bulk) */
     DOUBLE                    denmin;         /* min. dens. foam (opti.)*/
     DOUBLE                    denmax;         /* max. dens. foam (opti.)*/
     DOUBLE                    refdens;        /* reference density */
     DOUBLE                    oce;            /* exponent  */
     DOUBLE                    ocf;            /* factor    */
} MFOC;

/*----------------------------------------------------------------------*
 | closed cell metal foam material (optimization)            al 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _MFCC
{
     DOUBLE                    es;             /* Young's modulus (cell) */
     DOUBLE                    pr;             /* Possion ratio */
     DOUBLE                    dens;           /* density foam  */
     DOUBLE                    denss;          /* density (bulk) */
     DOUBLE                    denmin;         /* min. dens. foam (opti.)*/
     DOUBLE                    denmax;         /* max. dens. foam (opti.)*/
     DOUBLE                    refdens;        /* reference density */
     DOUBLE                    cce;            /* exponent  */
     DOUBLE                    ccf;            /* factor    */
} MFCC;
/*----------------------------------------------------------------------*
 | foam, closed cell, based on modified Neo Hook             al 4/01    |
 *----------------------------------------------------------------------*/
typedef struct _NHMFCC
{
     DOUBLE                    es;             /* Young's modulus (cell) */
     DOUBLE                    pr;             /* Possion ratio */
     DOUBLE                    dens;           /* density foam  */
     DOUBLE                    denss;          /* density (bulk) */
     DOUBLE                    denmin;         /* min. dens. foam (opti.)*/
     DOUBLE                    denmax;         /* max. dens. foam (opti.)*/
     DOUBLE                    refdens;        /* reference density */
     DOUBLE                    cce;            /* exponent  */
     DOUBLE                    ccf;            /* factor    */
} NHMFCC;
/*----------------------------------------------------------------------*
 | multi layer material  -> shell9                          sh 10/02    |
 | material, that can have differnt materials in different layers       |
 | in here is just the definition of the cross section                  |
 *----------------------------------------------------------------------*/
typedef struct _MULTI_LAYER
{
     INT                       num_klay;      /* number of kinematic layers */
     DOUBLE                   *klayhgt;       /* hgt of a kinematic layer in % of total thickness of the shell */
     struct _KINLAY           *kinlay;        /* one kinematic layer */

} MULTI_LAYER;
/*----------------------------------------------------------------------*
 | for multi layer material  -> shell9                      sh 10/02    |
 | information about one kinematic layer                                |
 *----------------------------------------------------------------------*/
typedef struct _KINLAY
{
     INT                       num_mlay;     /* number of material layer to this kinematic layer*/
     DOUBLE                   *mlayhgt;      /* hgt of a material layer in % of thickness of the adjacent kinematic layer*/
     INT                      *mmatID;       /* ID of multilayer material in every material layer */
     DOUBLE                   *phi;          /* rotation of the material in one material layer */
     INT                      *rot;          /* axis of rotation of the material: x=1, y=2, z=3*/
} KINLAY;
/*----------------------------------------------------------------------*
 | multilayer materials                                     sh 10/02    |
 | structure to hold all types of material laws                         |
 | is equivalent to the struct MATERIAL but is used for shell9 only     |
 *----------------------------------------------------------------------*/
typedef struct _MULTIMAT
{
     INT                       Id;           /* Id of the material */

     enum _MATERIAL_TYP        mattyp;       /* type of material */

     union
     {
     struct _STVENANT         *stvenant;     /* St. Venant-Kirchhoff material */
     struct _EL_ORTH          *el_orth;      /* linear elastic orthotropic material */
     struct _NEO_HOOKE        *neohooke;     /* Neo-Hooke material */
     struct _PL_MISES         *pl_mises;     /* von Mises material */
     struct _PL_HOFF          *pl_hoff;      /* anisotropic plastic material, based on hoffman-criterion */
     struct _PL_DP            *pl_dp;        /* Drucker Prager material */
     struct _PL_EPC           *pl_epc;       /* elastoplastic concrete material */
     }                         m;            /* union pointer to material specific structure */

} MULTIMAT;
/*----------------------------------------------------------------------*
 | linear elastic, orthotropic material                     sh 02/03    |
 *----------------------------------------------------------------------*/
typedef struct _ORTHOTROPIC
{
     DOUBLE                    emod1;
     DOUBLE                    emod2;
     DOUBLE                    emod3;
     DOUBLE                    gmod12;
     DOUBLE                    gmod13;
     DOUBLE                    gmod23;
     DOUBLE                    xnue12;
     DOUBLE                    xnue13;
     DOUBLE                    xnue23;
} ORTHOTROPIC;
/*----------------------------------------------------------------------*
 | interface elasto-damage-plasto surface material          ah 08/03    |
 *----------------------------------------------------------------------*/
typedef struct _IFMAT
{
     DOUBLE                    emod;
     DOUBLE                    kmod;
     DOUBLE                    gmod;
     DOUBLE                    dick;
     DOUBLE                    qmod;
     DOUBLE                    deltan;
     DOUBLE                    deltat;
     DOUBLE                    mu;
} IFMAT;
/*----------------------------------------------------------------------*
 | themodyn. based interface elasto-damage surface material   ah 09/04 |
 *----------------------------------------------------------------------*/
typedef struct _INTERF_THERM
{
     DOUBLE                    emod;
     DOUBLE                    nu;
     DOUBLE                    dick;
     INT                       equival;
     INT                       damtyp;
     DOUBLE                    kappa0_n;
     DOUBLE                    alpha_n;
     DOUBLE                    beta_n;
     DOUBLE                    kappa0_t;
     DOUBLE                    alpha_t;
     DOUBLE                    beta_t;
} INTERF_THERM;
/*----------------------------------------------------------------------*
 | isotropic damage material (mazars-pijadier-cabot)        ah 10/03    |
 *----------------------------------------------------------------------*/
typedef struct _DAM_MP
{
     DOUBLE                    youngs;
     DOUBLE                    nue;
     DOUBLE                    kappa_0;
     DOUBLE                    alpha;
     DOUBLE                    beta;
} DAM_MP;
/*----------------------------------------------------------------------*
 | isotropic gradient enhanced damage material              ah 10/03    |
 *----------------------------------------------------------------------*/
typedef struct _DAMAGE_GE
{
     INT                       equival;
     INT                       damtyp;
     DOUBLE                    crad;
     DOUBLE                    youngs;
     DOUBLE                    nue;
     DOUBLE                    kappa_0;
     DOUBLE                    kappa_m;
     DOUBLE                    alpha;
     DOUBLE                    beta;
     DOUBLE                    k_fac;
} DAMAGE_GE;
/*----------------------------------------------------------------------*
 | hyperelastic polyconvex material based on                   ah 10/07 |
 | Itskov                                                               |
 *----------------------------------------------------------------------*/
typedef struct _ITSKOV
{
     DOUBLE                    mu_GS;		/* material parameter ground substance*/
     DOUBLE                    mu_fibers;	/* material parameter fibers*/
     DOUBLE                    alpha;		/* material parameter fibers*/
     DOUBLE                    beta;		/* material parameter fibers*/
     DOUBLE                    epsilon; 	/* Penalty function */
     DOUBLE                    gamma;		/* Penalty function */
     DOUBLE                    comp;		/*desired incompressibility*/
     DOUBLE					   density;
} ITSKOV;
/*----------------------------------------------------------------------*
 | anisotropic hyperelastic polyconvex material based on      maf 11/07 |
 | Balzani et. al.                                                      |
 *----------------------------------------------------------------------*/
typedef struct _ANISOTROPIC_BALZANI
{
     DOUBLE                    c1;
     DOUBLE                    eps1;
     DOUBLE                    eps2;
     DOUBLE                    alpha1;
     DOUBLE                    alpha2;
     DOUBLE                    density;
     INT                       aloc;
     DOUBLE                    a1[3];
     DOUBLE                    alpha1_2;
     DOUBLE                    alpha2_2;
     DOUBLE                    a2[3];
} ANISOTROPIC_BALZANI;
/*----------------------------------------------------------------------*
 | Mooney-Rivlin material                                     maf 04/08 |
 *----------------------------------------------------------------------*/
typedef struct _MOONEYRIVLIN
{
     DOUBLE                    c1;
     DOUBLE                    c2;
     DOUBLE                    kap;
     DOUBLE                    lambda;
     DOUBLE                    density;
} MOONEYRIVLIN;
/*----------------------------------------------------------------------*
 | Yeoh material                                               tk 01/09 |
 *----------------------------------------------------------------------*/
typedef struct _YEOH
{
     DOUBLE                    c1;
     DOUBLE                    c2;
     DOUBLE                    c3;
     DOUBLE                    kap;
     DOUBLE                    density;
} YEOH;
/*----------------------------------------------------------------------*
 | Viscous NeoHookean material                                maf 05/08 |
 *----------------------------------------------------------------------*/
typedef struct _VISCONEOHOOKE
{
     DOUBLE                    youngs_slow;
     DOUBLE                    poisson;
     DOUBLE                    density;
     DOUBLE                    youngs_fast;
     DOUBLE                    relax;
     DOUBLE                    theta;
} VISCONEOHOOKE;
/*----------------------------------------------------------------------*
 | Viscous Anisotropic Fiber material                          tk 10/08 |
 *----------------------------------------------------------------------*/
typedef struct _VISCOANISOTROPIC
{
     DOUBLE                    kappa;
     DOUBLE                    mue;
     DOUBLE                    density;
     DOUBLE                    k1;
     DOUBLE                    k2;
     DOUBLE                    gamma;
     INT                       numstresstypes;
     DOUBLE                    beta[2];
     DOUBLE                    relax[2];
     INT                       tensonly;
} VISCOANISOTROPIC;
/*----------------------------------------------------------------------*
 | Continuum Chain Network Material Law                       maf 06/08 |
 *----------------------------------------------------------------------*/
typedef struct _CONTCHAINNETW
{
     DOUBLE                    lambda;
     DOUBLE                    mue;
     DOUBLE                    density;
     DOUBLE                    nchain;
     DOUBLE                    abstemp;
     DOUBLE                    contl_l;
     DOUBLE                    persl_a;
     DOUBLE                    r0;
     DOUBLE                    relax;
     INT                       initran;
     DOUBLE                    rembegt;
     INT                       updrate;
     DOUBLE                    difftol;
} CONTCHAINNETW;
/*----------------------------------------------------------------------*
 | Arterial Wall Model with Remodeling                        maf 07/08 |
 *----------------------------------------------------------------------*/
typedef struct _ARTWALLREMOD
{
     DOUBLE                    kappa;
     DOUBLE                    mue;
     DOUBLE                    density;
     DOUBLE                    k1;
     DOUBLE                    k2;
     DOUBLE                    gamma;
     INT                       init;
     DOUBLE                    rembegt;
     INT                       tensonly;
} ARTWALLREMOD;
/*----------------------------------------------------------------------*
 | Biological Cell model                                       rm 08/08 |
 *----------------------------------------------------------------------*/
typedef struct _BIOCELL
{
     DOUBLE                    density;
} BIOCELL;
/*----------------------------------------------------------------------*
 | Isotropic heat conduction coefficient                    bborn 03/06 |
 | of Fourier's law of heat conduction                                  |
 *----------------------------------------------------------------------*/
typedef struct _TH_FOURIER_ISO
{
     DOUBLE                    conduct;        /* heat conduction
                                                * coefficient [W/(m*K)] */
     DOUBLE                    capacity;       /* heat capacity
                                                * coefficient [J/(kg*K)] */
} TH_FOURIER_ISO;
/*----------------------------------------------------------------------*
 | General heat conduction coefficient matrix               bborn 04/06 |
 | of Fourier's law of heat conduction                                  |
 *----------------------------------------------------------------------*/
typedef struct _TH_FOURIER_GEN
{
     DOUBLE                    conduct[9];  /* heat conduction matrix
                                             * 3 rows with 3 param. are
                                             * stored consecutively in
                                             * a vector */
} TH_FOURIER_GEN;
/*----------------------------------------------------------------------*
 | Robinson's visco-plastic material                        bborn 03/07 |
 | material parameters                                                  |
 | [1] Butler, Aboudi and Pindera: "Role of the material constitutive   |
 |     model in simulating the reusable launch vehicle thrust cell      |
 |     liner response", J Aerospace Engrg, 18(1), 2005.                 |
 | [2] Arya: "Analytical and finite element solutions of some problems  |
 |     using a vsicoplastic model", Comput & Struct, 33(4), 1989.       |
 | [3] Arya: "Viscoplastic analysis of an experimental cylindrical      |
 |     thrust chamber liner", AIAA J, 30(3), 1992.                      |
 *----------------------------------------------------------------------*/
typedef struct _VP_ROBINSON
{
     enum VP_ROBINSON_KIND {                   /* kind of Robinson
                                                  material (slight
                                                  differences) */
       vp_robinson_kind_vague=0,               /* unset */
       vp_robinson_kind_butler,                /* Butler et al, 2005 [1] */
       vp_robinson_kind_arya,                  /* Arya, 1989 [2] */
       vp_robinson_kind_arya_narloyz,          /* Arya, 1992 [3] */
       vp_robinson_kind_arya_crmosteel         /* Arya, 1992 [3] */
     }                         kind;
     MAT_PARAM_MULT            youngmodul;     /* Young's modulus 'E' */
     DOUBLE                    possionratio;   /* Possion's ratio 'nu' */
     DOUBLE                    density;        /* material specific
						* weight 'rho' */
     DOUBLE                    thermexpans;    /* coefficient of thermal
						* expansion 'alpha' */
     DOUBLE                    hrdn_fact;      /* hardening factor 'A' */
     DOUBLE                    hrdn_expo;      /* hardening power 'n' */
     MAT_PARAM_MULT            shrthrshld;     /* Bingam-Prager shear
						* stress threshold 'K^2' */
     DOUBLE                    actv_tmpr;      /* activation temperature 'T_0' */
     DOUBLE                    actv_ergy;      /* activation energy 'Q_0' */
     DOUBLE                    m;              /* 'm' */
     DOUBLE                    g0;             /* 'G_0' */
     MAT_PARAM_MULT            beta;           /* 'beta' */
     MAT_PARAM_MULT            rcvry;          /* recovery factor 'R_0' */
     MAT_PARAM_MULT            h;              /* 'H' */
} VP_ROBINSON;

/*-------------------------------------------------------------------*
 | material parameters are calculated from microscale simulation     |
 |                                                         lw 06/07  |
 *-------------------------------------------------------------------*/
typedef struct _STRUCT_MULTISCALE
{
  INT            microdis;      /* Number of corresponding microscale
                                 * discretization */
  DOUBLE         V0;            /* initial volume of RVE (optional) */
  char   *micro_inputfile_name; /* inputfile name for microstructure */
} STRUCT_MULTISCALE;

/*----------------------------------------------------------------------*
 | material collection                                     gjb 07/08    |
 *----------------------------------------------------------------------*/
typedef struct _MATLIST
{
     INT         nummat; /* number of materials in list */
     INT*        matids; /* pointer to an array that holds the material ids */
} MATLIST;

/*----------------------------------------------------------------------*
 | material parameters for ion species in electrlyte solution gjb 07/08 |
 *----------------------------------------------------------------------*/
typedef struct _ION
{
     INT         valence;     /* valence (= charge number) */
     DOUBLE      diffusivity; /* kinematic diffusivity     */
} ION;
/*----------------------------------------------------------------------*
 | material parameters constant 1D_Artery                     ism 06/09 |
 *----------------------------------------------------------------------*/
typedef struct _CNST_ART
{
     DOUBLE                    viscosity;      /* Newtonian viscosity of blood */
     DOUBLE                    density;        /* material specific weight */
     DOUBLE                    youngs;         /* Young's modulus at the Inlet*/
     DOUBLE                    possionratio;   /* Possion ratio */
     DOUBLE                    diam;           /* initial diameter of an artery*/
     DOUBLE                    th;             /* artery thickness */
     DOUBLE                    pext1;          /* artery fixed external pressure 1 */
     DOUBLE                    pext2;          /* artery fixed external pressure 2 */

} CNST_ART;
/*----------------------------------------------------------------------*
 | Anisotropic Fiber material for arteries                     st 11/09 |
 *----------------------------------------------------------------------*/
typedef struct _HOLZAPFELCARDIO
{
     DOUBLE                    kappa;
     DOUBLE                    mue;
     DOUBLE                    density;
     DOUBLE                    k1;
     DOUBLE                    k2;
     DOUBLE                    gamma;
     DOUBLE                    minstretch;
} HOLZAPFELCARDIO;
/*----------------------------------------------------------------------*
 | Anisotropic material for arteries cf Humphrey               st 11/09 |
 *----------------------------------------------------------------------*/
typedef struct _HUMPHREYCARDIO
{
     DOUBLE                    kappa;
     DOUBLE                    mue;
     DOUBLE                    density;
     DOUBLE                    k1c;
     DOUBLE                    k2c;
     DOUBLE                    k1m;
     DOUBLE                    k2m;
     DOUBLE                    phie;
     DOUBLE                    phic;
     DOUBLE                    phim;
} HUMPHREYCARDIO;

#endif /* defined(CCADISCRET) && !defined(D_SHELL8) */
