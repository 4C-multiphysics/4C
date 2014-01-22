/*!
\file scatra_ele_impl_utils.cpp

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
*/
/*----------------------------------------------------------------------*/

#include "scatra_ele_impl_utils.H"
#include "../drt_lib/standardtypes_cpp.H"
#include "../drt_lib/drt_condition_utils.H"

namespace SCATRA
{

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool IsBinaryElectrolyte(const std::vector<double>& valence)
{
  int numions(0);
  for (size_t k=0; k < valence.size(); k++)
  {
    if (abs(valence[k]) > EPS10)
      numions++;
  }
  return (numions == 2);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<int> GetIndicesBinaryElectrolyte(const std::vector<double>& valence)
{
  // indices of the two charged species to be determined
  std::vector<int> indices;
  for (size_t k=0; k < valence.size(); k++)
  {
    // is there some charge?
    if (abs(valence[k]) > EPS10)
      indices.push_back(k);
  }
  if (indices.size() != 2) dserror("Found no binary electrolyte!");

  return indices;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
double CalResDiffCoeff(
    const std::vector<double>& valence,
    const std::vector<double>& diffus,
    const std::vector<int>& indices
    )
{
  if (indices.size() != 2) dserror("Non-matching number of indices!");
  const int first = indices[0];
  const int second = indices[1];
  if ((valence[first]*valence[second])>EPS10)
    dserror("Binary electrolyte has no opposite charges.");
  const double n = ((diffus[first]*valence[first])-(diffus[second]*valence[second]));
  if (abs(n) < EPS12)
    dserror("denominator in resulting diffusion coefficient is nearly zero");

  return diffus[first]*diffus[second]*(valence[first]-valence[second])/n;
}


/*-------------------------------------------------------------------------------*
 |find elements of inflow section                                rasthofer 01/12 |
 |for turbulent low Mach number flows with turbulent inflow condition            |
 *-------------------------------------------------------------------------------*/
bool InflowElement(const DRT::Element* ele)
{
  bool inflow_ele = false;

  std::vector<DRT::Condition*> myinflowcond;

  // check whether all nodes have a unique inflow condition
  DRT::UTILS::FindElementConditions(ele, "TurbulentInflowSection", myinflowcond);
  if (myinflowcond.size()>1)
    dserror("More than one inflow condition on one node!");

  if (myinflowcond.size()==1)
   inflow_ele = true;

  return inflow_ele;
}


} // namespace SCATRA

