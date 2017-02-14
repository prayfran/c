/*
 * testRadix.c - simple driver to test methods of radix.h.
 *
 * "Copyright (c) 2013 by Fritz Sieker."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software.
 *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE AUTHOR
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
 * BASIS, AND THE AUTHOR NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "radix.h"

/** @mainpage cs270 Fall 2013 Programming Assignment ? - radix conversion in C
 *  \htmlinclude "RADIX.html"
 */

/** @file: testRadix.c
 *  @brief Driver to test functions of radix.c (do not modify)
 *
 * @details This is a driver program to test the functions
 * defined in radix.h and implemented in radix.c. The program takes three
 * command line parameters and calls one of the methods, then
 * prints the results. To see how to use the program, execute
 * <code>testRadix</code> in a terminal window. This will print a usage
 * statement defining how to run the program. The first parameter of the
 * program is always a key defining which function to run.
 * options are:
 * <ul>
 * <li><b>c2i</b> convert a character to its ordinal value</li>
 * <li><b>i2c</b> convert a value to its character representation</li>
 * <li><b>s2i</b> convert a string of characters to a value</li>
 * <li><b>i2s</b> print the digit(s) of a value</li>
 * </ul>
 * <p>
 * The 2nd parameter is always the radix you work in (2..36). The 3rd
 * parameter will be either a number (base 10) when you are converting an
 * number to a <b>character/characters</b>. It will be one/multiple characters
 * when converting to a <b>number</b>.
 * <p>
 * A sample execution might be: <code>testRadix c2i 16 A</code>
 * <p>
 * All values may be entered as unsigned decimal numbers
 * <p>
 * @author Fritz Sieker
 */

/** Print a usage statement, then exit the program returning a non zero
 * value, the Linux convention indicating an error
 */
static void usage() {
  puts("Usage: testRadix c2i radix digit");
  puts("       testRadix i2c radix value");
  puts("       testRadix s2i radix");
  puts("       testRadix i2s radix");
  puts("       testRadix s2fr radix");
  exit(1);
}

/** Entry point of the program
 * @param argc count of arguments, will always be at least 1
 * @param argv array of parameters to program argv[0] is the name of
 * the program, so additional parameters will begin at index 1.
 * @return 0 the Linux convention for success.
 */
int main (int argc, char* argv[]) {
  if (argc < 3)
    usage();
  else if (argc > 4)
    usage();
  
  char* op    = argv[1];
  int   radix = atoi(argv[2]);
  if (((strcmp(op, "c2i") == 0) || (strcmp(op, "i2c") == 0)) && argc == 3)
    usage();

  if (strcmp(op, "c2i") == 0) {
    printf("%d", char2int(radix, argv[3][0]));
  }

  else if (strcmp(op, "i2c") == 0) {
    printf("%c", int2char(radix, atoi(argv[3])));
  }


  else if (((strcmp(op, "s2i") == 0) ||
	    (strcmp(op, "i2s") == 0) ||
	    (strcmp(op, "s2fr") == 0))
	   && argc == 4)
    usage();

  else if (strcmp(op, "s2i") == 0) {
    printf("Please enter a radix %d number: ", radix);
    printf("%d", str2int(radix));
    printf("\n");
  }

  else if (strcmp(op, "i2s") == 0) {
    int value;
    printf("Please enter an integer value: ");
    scanf("%d", &value);
    int2str(radix, value);
  }


  else if (strcmp(op, "s2fr") == 0) {
    printf("Please enter a radix %d fraction: 0.", radix);
    printf("%f", str2frac(radix));
    printf("\n");
  }

  else
    usage();
  
  printf("\n");
  return 0;
}

