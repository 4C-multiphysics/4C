for (int vi=0; vi<iel; ++vi)
{
    for (int ui=0; ui<iel; ++ui)
    {
   
    /* standard Galerkin terms: */
    /* convective term */
    estif_(vi*3, ui*3)         += timefacfac*funct_(vi)*(conv_c_(ui) + conv_r_(0, 0, ui)) ;
    estif_(vi*3, ui*3 + 1)     += timefacfac*funct_(vi)*conv_r_(0, 1, ui) ;
    estif_(vi*3 + 1, ui*3)     += timefacfac*funct_(vi)*conv_r_(1, 0, ui) ;
    estif_(vi*3 + 1, ui*3 + 1) += timefacfac*funct_(vi)*(conv_c_(ui) + conv_r_(1, 1, ui)) ;

    /* viscous term */
    estif_(vi*3, ui*3)         += fac*time2nue*derxy_(0, ui)*derxy_(0, vi) +
	                          nu_*timefacfac*derxy_(1, ui)*derxy_(1, vi) ;
    estif_(vi*3, ui*3 + 1)     += nu_*timefacfac*derxy_(0, ui)*derxy_(1, vi) ;
    estif_(vi*3 + 1, ui*3)     += nu_*timefacfac*derxy_(0, vi)*derxy_(1, ui) ;
    estif_(vi*3 + 1, ui*3 + 1) += fac*time2nue*derxy_(1, ui)*derxy_(1, vi) +
	                          nu_*timefacfac*derxy_(0, ui)*derxy_(0, vi) ;

    /* pressure term */
    estif_(vi*3, ui*3 + 2)     += -(timefacfac*funct_(ui)*derxy_(0, vi)) ;
    estif_(vi*3 + 1, ui*3 + 2) += -(timefacfac*funct_(ui)*derxy_(1, vi)) ;

    /* transient term */
    estif_(vi*3, ui*3)         += fac*funct_(ui)*funct_(vi) ;
    estif_(vi*3 + 1, ui*3 + 1) += fac*funct_(ui)*funct_(vi) ;

    /* continuity term */
    estif_(vi*3 + 2, ui*3)     += timefacfac*funct_(vi)*derxy_(0, ui) ;
    estif_(vi*3 + 2, ui*3 + 1) += timefacfac*funct_(vi)*derxy_(1, ui) ;


    /* stabilization terms: */
    /* convective stabilization: */
    /* convective term */
    estif_(vi*3, ui*3)         += ttimetauM*(conv_c_(ui)*conv_c_(vi) +
					     conv_c_(vi)*conv_r_(0, 0, ui) +
					     velint_(0)*derxy_(0, vi)*conv_r_(0, 0, ui) +
					     velint_(1)*derxy_(0, vi)*conv_r_(0, 1, ui)) ;
    estif_(vi*3, ui*3 + 1)     += ttimetauM*(conv_c_(vi)*conv_r_(0, 1, ui) +
					     velint_(0)*derxy_(1, vi)*conv_r_(0, 0, ui) +
					     velint_(1)*derxy_(1, vi)*conv_r_(0, 1, ui)) ;
    estif_(vi*3 + 1, ui*3)     += ttimetauM*(conv_c_(vi)*conv_r_(1, 0, ui) +
					     velint_(0)*derxy_(0, vi)*conv_r_(1, 0, ui) +
					     velint_(1)*derxy_(0, vi)*conv_r_(1, 1, ui)) ;
    estif_(vi*3 + 1, ui*3 + 1) += ttimetauM*(conv_c_(ui)*conv_c_(vi) +
					     conv_c_(vi)*conv_r_(1, 1, ui) +
					     velint_(0)*derxy_(1, vi)*conv_r_(1, 0, ui) +
					     velint_(1)*derxy_(1, vi)*conv_r_(1, 1, ui)) ;

    /* viscous term */
    estif_(vi*3, ui*3)         += -2.0*nu_*ttimetauM*(-(conv_c_(vi)*viscs2_(0, 0, ui)) +
						      funct_(ui)*derxy_(0, vi)*visc_old_(0)) ;
    estif_(vi*3, ui*3 + 1)     += -2.0*nu_*ttimetauM*(-(conv_c_(vi)*viscs2_(0, 1, ui)) +
						      funct_(ui)*derxy_(1, vi)*visc_old_(0)) ;
    estif_(vi*3 + 1, ui*3)     += -2.0*nu_*ttimetauM*(-(conv_c_(vi)*viscs2_(0, 1, ui)) +
						      funct_(ui)*derxy_(0, vi)*visc_old_(1)) ;
    estif_(vi*3 + 1, ui*3 + 1) += -2.0*nu_*ttimetauM*(-(conv_c_(vi)*viscs2_(1, 1, ui)) +
						      funct_(ui)*derxy_(1, vi)*visc_old_(1)) ;

    /* pressure term */
    estif_(vi*3, ui*3)         += ttimetauM*funct_(ui)*gradp_(0)*derxy_(0, vi) ;
    estif_(vi*3, ui*3 + 1)     += ttimetauM*funct_(ui)*gradp_(0)*derxy_(1, vi) ;
    estif_(vi*3, ui*3 + 2)     += ttimetauM*conv_c_(vi)*derxy_(0, ui) ;
    estif_(vi*3 + 1, ui*3)     += ttimetauM*funct_(ui)*gradp_(1)*derxy_(0, vi) ;
    estif_(vi*3 + 1, ui*3 + 1) += ttimetauM*funct_(ui)*gradp_(1)*derxy_(1, vi) ;
    estif_(vi*3 + 1, ui*3 + 2) += ttimetauM*conv_c_(vi)*derxy_(1, ui) ;

    /* transient term */
    estif_(vi*3, ui*3)         += timetauM*funct_(ui)*(conv_c_(vi) + velint_(0)*derxy_(0, vi)) ;
    estif_(vi*3, ui*3 + 1)     += timetauM*funct_(ui)*velint_(0)*derxy_(1, vi) ;
    estif_(vi*3 + 1, ui*3)     += timetauM*funct_(ui)*velint_(1)*derxy_(0, vi) ;
    estif_(vi*3 + 1, ui*3 + 1) += timetauM*funct_(ui)*(conv_c_(vi) + velint_(1)*derxy_(1, vi)) ;

    /* viscous stabilization: */
    /* convective term */
    /*estif_(vi*3, ui*3)         += 2.0*nu_*ttimetauMp*(conv_c_(ui)*viscs2_(0, 0, vi) +
						      viscs2_(0, 0, vi)*conv_r_(0, 0, ui) +
						      viscs2_(0, 1, vi)*conv_r_(1, 0, ui)) ;
    estif_(vi*3, ui*3 + 1)     += 2.0*nu_*ttimetauMp*(conv_c_(ui)*viscs2_(0, 1, vi) +
						      viscs2_(0, 0, vi)*conv_r_(0, 1, ui) +
						      viscs2_(0, 1, vi)*conv_r_(1, 1, ui)) ;
    estif_(vi*3 + 1, ui*3)     += 2.0*nu_*ttimetauMp*(conv_c_(ui)*viscs2_(0, 1, vi) +
						      viscs2_(0, 1, vi)*conv_r_(0, 0, ui) +
						      viscs2_(1, 1, vi)*conv_r_(1, 0, ui)) ;
    estif_(vi*3 + 1, ui*3 + 1) += 2.0*nu_*ttimetauMp*(conv_c_(ui)*viscs2_(1, 1, vi) +
						      viscs2_(0, 1, vi)*conv_r_(0, 1, ui) +
						      viscs2_(1, 1, vi)*conv_r_(1, 1, ui)) ;*/

    /* viscous term */
    /*estif_(vi*3, ui*3)         += 4.0*(nu_*nu_)*ttimetauMp*(viscs2_(0, 0, ui)*viscs2_(0, 0, vi) +
							    viscs2_(0, 1, ui)*viscs2_(0, 1, vi)) ;
    estif_(vi*3, ui*3 + 1)     += 4.0*(nu_*nu_)*ttimetauMp*(viscs2_(0, 0, vi)*viscs2_(0, 1, ui) +
							    viscs2_(0, 1, vi)*viscs2_(1, 1, ui)) ;
    estif_(vi*3 + 1, ui*3)     += 4.0*(nu_*nu_)*ttimetauMp*(viscs2_(0, 0, ui)*viscs2_(0, 1, vi) +
							    viscs2_(0, 1, ui)*viscs2_(1, 1, vi)) ;
    estif_(vi*3 + 1, ui*3 + 1) += 4.0*(nu_*nu_)*ttimetauMp*(viscs2_(0, 1, ui)*viscs2_(0, 1, vi) +
							    viscs2_(1, 1, ui)*viscs2_(1, 1, vi)) ;*/

    /* pressure term */
    /*estif_(vi*3, ui*3 + 2)     += 2.0*nu_*ttimetauMp*(derxy_(0, ui)*viscs2_(0, 0, vi) +
						      derxy_(1, ui)*viscs2_(0, 1, vi)) ;
    estif_(vi*3 + 1, ui*3 + 2) += 2.0*nu_*ttimetauMp*(derxy_(0, ui)*viscs2_(0, 1, vi) +
						      derxy_(1, ui)*viscs2_(1, 1, vi)) ;*/

    /* transient term */
    /*estif_(vi*3, ui*3)         += tau_Mp*time2nue*funct_(ui)*viscs2_(0, 0, vi) ;
    estif_(vi*3, ui*3 + 1)     += tau_Mp*time2nue*funct_(ui)*viscs2_(0, 1, vi) ;
    estif_(vi*3 + 1, ui*3)     += tau_Mp*time2nue*funct_(ui)*viscs2_(0, 1, vi) ;
    estif_(vi*3 + 1, ui*3 + 1) += tau_Mp*time2nue*funct_(ui)*viscs2_(1, 1, vi) ;*/

    /* pressure stabilization: */
    /* convective term */
    estif_(vi*3 + 2, ui*3)     += ttimetauMp*(conv_c_(ui)*derxy_(0, vi) +
					      derxy_(0, vi)*conv_r_(0, 0, ui) +
					      derxy_(1, vi)*conv_r_(1, 0, ui)) ;
    estif_(vi*3 + 2, ui*3 + 1) += ttimetauMp*(conv_c_(ui)*derxy_(1, vi) +
					      derxy_(0, vi)*conv_r_(0, 1, ui) +
					      derxy_(1, vi)*conv_r_(1, 1, ui)) ;

    /* viscous term */
    estif_(vi*3 + 2, ui*3)     += 2.0*nu_*ttimetauMp*(derxy_(0, vi)*viscs2_(0, 0, ui) +
						      derxy_(1, vi)*viscs2_(0, 1, ui)) ;
    estif_(vi*3 + 2, ui*3 + 1) += 2.0*nu_*ttimetauMp*(derxy_(0, vi)*viscs2_(0, 1, ui) +
						      derxy_(1, vi)*viscs2_(1, 1, ui)) ;

    /* pressure term */
    estif_(vi*3 + 2, ui*3 + 2) += ttimetauMp*(derxy_(0, ui)*derxy_(0, vi) +
					      derxy_(1, ui)*derxy_(1, vi)) ;

    /* transient term */
    estif_(vi*3 + 2, ui*3)     += timetauMp*funct_(ui)*derxy_(0, vi) ;
    estif_(vi*3 + 2, ui*3 + 1) += timetauMp*funct_(ui)*derxy_(1, vi) ;

    /* continuity stabilization: */
    estif_(vi*3, ui*3)         += (thsl*thsl)*tau_C*derxy_(0, ui)*derxy_(0, vi) ;
    estif_(vi*3, ui*3 + 1)     += (thsl*thsl)*tau_C*derxy_(0, vi)*derxy_(1, ui) ;
    estif_(vi*3 + 1, ui*3)     += (thsl*thsl)*tau_C*derxy_(0, ui)*derxy_(1, vi) ;
    estif_(vi*3 + 1, ui*3 + 1) += (thsl*thsl)*tau_C*derxy_(1, ui)*derxy_(1, vi) ;

    /* additional stabilization term for Newton iteration: */
    estif_(vi*3, ui*3)         += -(timetauM*funct_(ui)*rhsint_(0)*derxy_(0, vi)) ;
    estif_(vi*3, ui*3 + 1)     += -(timetauM*funct_(ui)*rhsint_(0)*derxy_(1, vi)) ;
    estif_(vi*3 + 1, ui*3)     += -(timetauM*funct_(ui)*rhsint_(1)*derxy_(0, vi)) ;
    estif_(vi*3 + 1, ui*3 + 1) += -(timetauM*funct_(ui)*rhsint_(1)*derxy_(1, vi)) ;

  }
}
