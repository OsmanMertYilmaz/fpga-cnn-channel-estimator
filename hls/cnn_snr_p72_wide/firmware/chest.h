#ifndef CHEST_H_
#define CHEST_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void chest(
    hls::stream<input_t> &h_ls,
    hls::stream<result_t> &layer16_out
);

// hls-fpga-machine-learning insert emulator-defines


#endif
