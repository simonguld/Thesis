#ifndef WRITE_HPP_
#define WRITE_HPP_

#include "serialization.hpp"

/** Saves the complete frame of the system */
void WriteFrame(unsigned t);
/** Saves the complete frame of the system (old style)*/
void WriteFrame_old(unsigned t);
/** Write the simulation parameters */
void WriteParams();
/** Ask and maybe delete output files
 *
 * We do not overwrite any file! This is a mean of protection for light headed
 * grad students.
 * */
void ClearOutput();
/** Create temporary directory (in /tmp/ by default)
 *
 * The tmp directory used for output is /tmp/bunch-of-numbers, where bunch-
 * of-numbers is a bunch of numbers that has beens generated using a hash
 * function from the run name and a random salt.
 * */
void CreateOutputDir();

#endif//WRITE_HPP_
