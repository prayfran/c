#include "field.h"
#include "flt32.h"

/** @file flt32.c
 *  @brief You will modify this file and implement nine functions
 *  @details Your implementation of the functions defined in flt32.h.
 *  You may add other function if you find it helpful. Added function
 *  should be declared <b>static</b> to indicate they are only used
 *  within this file.
 *  <p>
 *  @author <b>Your name</b> goes here
 */

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
int flt32_get_sign (flt32 x) {


int sign = getField(x, 31, 31, 0);

if(sign == 1){
	return 1;
	}
	else
		{
		return 0;
		}
			
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
int flt32_get_exp (flt32 x) {

int exp  = getField(x, 30, 23, 0);

  return exp; 
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
int flt32_get_val (flt32 x) {
if(x==0)
{return 0;
}
int frac = getField(x,  22,  0, 0)+(1<<23);

  return frac;
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
void flt32_get_all(flt32 x, int* sign, int*exp, int* val) {
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
int flt32_left_most_1 (int bits) {
int i =0;
for(i=31;i>=0;i--)
{
   if((bits&(1<<i))!=0)
	{
	break;
	}
}
  return i;
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
flt32 flt32_abs (flt32 x) {
  return 0x7FFFFFFF & x;
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
flt32 flt32_negate (flt32 x) {
	if(x < 0){
	x=setField(x,31,31,0);
		}else{
	x=setField(x,31,31,1);
	}
  return x;
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
flt32 flt32_add (flt32 x, flt32 y) {
int signx = flt32_get_sign(x);
int signy = flt32_get_sign(y);
int xexp = flt32_get_exp(x);
int yexp = flt32_get_exp(y);
int xval = flt32_get_val(x);
int yval = flt32_get_val(y);
int newval;
int newvalsign =0;
int newvalexp;
int new32bit = 0;
if(x == 0 && y == 0)
{
return 0;
}
if(xexp < yexp)
{
	xval>>=(yexp-xexp);
	xexp = yexp;
}
if(yexp<xexp)
{
	yval>>=(xexp-yexp);
	yexp = xexp;
}
newvalexp = xexp;
	if(signx == 1 )
	{
		xval = -xval;
	}
	if(signy == 1)
	{
		yval = -yval;
	}

	newval = xval + yval;

	if(flt32_get_sign(newval)== 1)
	{
		newvalsign = 1;
 		newval = -newval;
	}

	
	int leftmost = flt32_left_most_1(newval);

	if(leftmost> 23)
	{
		 newval>>=(leftmost- 23);
		 newvalexp +=  (leftmost- 23);
	}

	if(flt32_left_most_1(newval)< 23)
	{
 	newval<<=(23 - leftmost);
	newvalexp -= (23 - leftmost);
	}

	new32bit=setField (new32bit, 31, 31, newvalsign);
	new32bit=setField (new32bit, 30, 23, newvalexp);
	new32bit=setField (new32bit, 22, 0, newval);

	return new32bit;
}

/** @todo Implement in flt32.c based on documentation contained in flt32.h */
flt32 flt32_sub (flt32 x, flt32 y) {
if(x==y){
return 0;
}else{
  return flt32_add(x,flt32_negate(y)); 
}
}
