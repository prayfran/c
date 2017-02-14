#ifndef __RADIX_H__
#define __RADIX_H__

/*
 * rdixield.h - simple radix conversion functions to get students familiar
 *           with C with a <b>very</b> brief introduction to strings.
 *
 * "Copyright (c) 2013 by Fritz Sieker."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written
 * agreement is hereby granted, provided that the above copyright notice
 * and the following two paragraphs appear in all copies of this software,
 * that the files COPYING and NO_WARRANTY are included verbatim with
 * any distribution, and that the contents of the file README are included
 * verbatim as part of a file named README with any distribution.
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

/** @file radix.h
 *  @brief Defines interface of radix.c functions (do not modify)
 *  @details This file defines the interface to a C file radix.c that
 *  you will complete.
 *  <p>
 *
 *  Numbers can be reperesented in various number bases. Humans use
 *  base 10.  Many times it is convenient to work in other bases. For
 *  bases with a radix greater that 10, other characters must be used
 *  to represent the "digits" of the human readable form.  The
 *  characters '0' thru '9' represnt values 0 to 9.  The characters
 *  'A' to 'Z' represent values greater that 10 with 'A' representing
 *  10, 'B' representing 11 and so on up to 'Z' representing 35.  We
 *  will not wory about number bases beyond that.  Thus, only '0'..'9'
 *  and 'A'..'Z' will ever be used.  For convenience, the characters
 *  'a'..'z' are also allowed, and each lower case letter represents
 *  the same as its upper case equivalent.
 *
 *  <p>
 *  @author Fritz Sieker, modified by Sanjay Rajopadhye
 */

/** Get the character associated with a value in a specified radix
 *  @param radix the radix in which you are working (2-36)
 *  @param value should be in range 0 to radix - 1
 *  @return the character '?' if the value is outside the radix range,
 *  otherwise return the character that reprensets that digit.  For a
 *  radix greater that 10, use the uppercase letters 'A' to 'Z' to
 *  represent digits beyond a value of 9.
 */
char int2char (int radix, int value);

/** Convert a character representing a digit in the specified base to
    its value
 *  @param radix the radix in which you are working (2-36)
 *  @param digit the character to convert
 *  @return -1 if the digit is not legal for the specified base, otherwise
 *  return the the correct value (0 to radix -1).
 */
int char2int (int radix, char digit);

/** Print the sequence of characters/digits that represents, in the
 *  specified radix, the given integer.  <b>Your program cannot use
 *  arays or strings in C (even if know about them).  It should
 *  determine the characters to print out using the algorithm of
 *  repeated division</b>.  The only output function you are allowed
 *  to use is <code>putchar()</code> from <code>stdio.h</code>.
 *  [Hint: think recursion].
 *  @param radix the radix in which you are working (2-36)
 *  @param value the value to print out.
 */
void int2str (int radix, int value);

/** Convert a sequence of characters representing a number in the
    given radix into the integer value that it represents.  The only
    input function you are allowed to use is <code>getchar()</code>
    from <code>stdio.h</code>.
 *  @param radix the radix in which you are working (2-36)
 *  @return -1 if the value contains an illegal digit for the radix, otherwise
 *  return the value.
 */
int str2int (int radix);

/** Convert a sequence of characters representing a fractional number
    in the given radix into the integer value that it represents.  The
    only input function you are allowed to use is
    <code>getchar()</code> from <code>stdio.h</code>.  <b>This
    function is for extra credit</b>
 *  @param radix the radix in which you are working (2-36)
 *  @return -1 if the value contains an illegal digit for the radix, otherwise
 *  return the value.
 */
double str2frac (int radix);

#endif

