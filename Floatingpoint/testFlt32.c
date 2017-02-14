/*
 * testFlt32.c - simple driver to test methods of flt32.h.
 *
 * "Copyright (c) 2012 by Fritz Sieker."
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
#include "field.h"
#include "flt32.h"

/** @mainpage cs270 Fall 2013 Programming Assignment ? - Floating Point math
 *  \htmlinclude "FLOAT.html"
 */

/** @file: testFlt32.c
 *  @brief Driver to test functions of flt32.c (do not modify)
 *
 * @details This is a driver program to test the functions
 * defined in flt32.h and implemented in flt32.c. The program takes one
 * or more command line parameters and calls one of the methods, then
 * prints the results. To see how to use the program, execute
 * <code>testFlt32</code> in a terminal window. This will print a usage
 * statement defining how to run the program. The first parameter of the
 * program is always a key defining which function to run. The
 * options are:
 * <ul>
 * <li><b>bin</b> print the next parameter in decimal, hex and binary</li>
 * <li><b>get</b> extract a field from a value (4 more parameter)</li>
 * <li><b>set</b> set a field in a value (4 more parameters)</li>
 * </ul>
 * <p>
 * A sample execution might be: <code>testField get 0xABCD 9 4 0</code>
 * <p>
 * which prints <pre><code>dec: 60  hex: 0x3C  bin: 0000-0000-0000-0000-0000-0000-0011-1100</code></pre>
 * <p>
 * All values may be entered as signed decimal numbers or as hex values
 * by beginning it with <code>0x</code>.
 * <p>
 * @author Fritz Sieker
 */

/** Print the binary representation of a value starting at the specified
 * bit position. A separator is printed every 4 bits for easy reading.
 * @param value the value to be printed
 * @param msb the bit position to begin printing (31 to 0)
 */
void printBinaryMSB (int value, int msb) {
  while (msb >= 0) {
    putchar(((value & (1 << msb)) ? '1' : '0'));
    if (msb && ((msb & 0x3) == 0))
      putchar('-');
    msb--;
  }
}

/** Print a 32 bit binary representation of a value.
 * @param value the value to be printed
 */
void printBinary (int value) {
  printBinaryMSB(value, 31);
}

/** Print a usage statement, then exit the program returning a non zero
 * value, the Linux convention indicating an error
 */
static void usage() {
  puts("usage: testField abs FPvalue");
  puts("       testField all FPvalue");
  puts("       testField add FPvalue1 FPvalue2");
  puts("       testField bin FPvalue");
  puts("       testField exp FPvalue");
  puts("       testField lm1 Intvalue");
  puts("       testField neg FPvalue");
  puts("       testField sign FPvalue");
  puts("       testField sub FPvalue1 FPvalue2");
  puts("       testField val FPvalue");
  exit(1);
}

/** print the value in decimal, hex and binary.
 * @param result the value to be printed.
 */
static void print_result (int result) {
  printf("dec: %d  hex: 0x%X  bin: ", result, result);
  printBinary(result);
}

/** Entry point of the program
 * @param argc count of arguments, will always be at least 1
 * @param argv array of parameters to program argv[0] is the name of
 * the program, so additional parameters will begin at index 1.
 * @return 0 the Linux convention for success.
 */
int main (int argc, char* argv[]) {
  union {
    float f;
    int   i;
  } u;

  int x, y;

  if (argc < 3) {
    usage();
  }

  char* op = argv[1];
  u.f = (float) atof(argv[2]);
  x = u.i;

  if (argc > 3) {
    u.f = (float) atof(argv[3]);
    y = u.i;
  }

  if ((strcmp(op, "abs") == 0) && (argc == 3)){
    u.i = flt32_abs(x);
    printf("%f\n", u.f);
  }

  else if ((strcmp(op, "all") == 0) && (argc == 3)){
    int sign, exp, val;
    flt32_get_all(x, &sign, &exp, &val);
    printf("sign: %d exp: %d val: %d\n", sign, exp, val);
  }

  else if ((strcmp(op, "add") == 0) && (argc == 4)) {
    u.i = flt32_add(x, y);
    printf("%f\n", u.f);
  }

  else if ((strcmp(op, "bin") == 0) && (argc == 3))  {
    print_result(x);
    puts("\n");
  }

  else if ((strcmp(op, "exp") == 0) && (argc == 3)) {
    printf("%d\n", flt32_get_exp(x));
  }

  else if ((strcmp(op, "lm1") == 0) && (argc == 3)) {
    printf("%d\n", flt32_left_most_1(x));
  }

  else if ((strcmp(op, "neg") == 0) && (argc == 3)) {
    u.i = flt32_negate(x);
    printf("%f\n", u.f);
  }

  else if ((strcmp(op, "sign") == 0) && (argc == 3)) {
    printf("%d\n", flt32_get_sign(x));
  }

  else if ((strcmp(op, "sub") == 0) && (argc == 4)) {
    u.i = flt32_sub(x, y);
    printf("%f\n", u.f);
  }

  else if ((strcmp(op, "val") == 0) && (argc == 3)) {
    print_result(flt32_get_val(x));
    puts("\n");
  }

  else
    usage();
  
  printf("\n");
  return 0;
}
