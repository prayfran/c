#include<stdio.h>
#include<string.h>

#include "radix.h"

/** @file radix.c
 *  @brief You will modify this file and implement three or four functions
 *  @details Your implementation of the functions defined in radix.h.
 * <p>
 * @author <b>Your name</b> goes here
 */

/** @todo Implement in radix.c based on documentation contained in radix.h */
char int2char (int radix, int value) {
char newval;

if(value <= 9)
{
	newval= value +'0';
}
else
{
	newval = value - 10 + 'A';
}

if(value < 0||value > radix||value>'Z'|| radix <2 || radix > 36){

  return '?';

}else{

return newval;

}
}

/** @todo Implement in radix.c based on documentation contained in radix.h */
int char2int (int radix, char digit) {
int value;


if(digit <= '9')
{
	value = digit - '0';
}
else
{
if(digit >='a' && digit <= 'z')
	{
	int newvalue = digit - 32;
	value = newvalue - 'A' + 10;
	}else{
	value = digit - 'A' + 10;}
}
if((value >= radix) || (digit>'z') || (digit>'9' && digit<'A') || (digit>'Z' && digit<'a') || radix < 2 || radix >36)
   {
	return -1;
   }
  return value;

}

void int2str (int radix, int value) {

int superevil = value;
int c;
int a;
char b;
if(superevil != 0)

	{

	c = superevil%radix;

	a = superevil/radix;

	b = int2char(radix,c);
	
	int2str(radix,a);

	putchar(b);

}

}


int str2int (int radix) {

char cd;
int b = 0;
int r = 0;

if(radix<2 && radix >36) return -1;
do
{
	cd=getchar();
if((cd < '0' || cd > '9')&& (cd <'A' || cd>'Z') && (cd < 'a' || cd > 'z') ) break;
	r *= radix;
	b=char2int(radix,cd);
	
	r += b;
			
			
}
while((cd != '\n'));
return r;
 

}
/** @todo Implement in radix.c based on documentation contained in radix.h */
double str2frac (int radix) {
  return -1.0;
}





