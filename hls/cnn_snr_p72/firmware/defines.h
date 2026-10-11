#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<12,4,AP_RND_CONV,AP_SAT,0>, 3*1> input_t;
typedef nnet::array<ap_fixed<12,4,AP_RND_CONV,AP_SAT,0>, 3*1> layer18_t;
typedef ap_fixed<32,14> conv1_accum_t;
typedef nnet::array<ap_fixed<32,14>, 16*1> layer2_t;
typedef ap_fixed<8,1> weight2_t;
typedef ap_fixed<8,1> bias2_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer4_t;
typedef ap_fixed<18,8> conv1_relu_table_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer19_t;
typedef ap_fixed<32,14> conv2_accum_t;
typedef nnet::array<ap_fixed<32,14>, 16*1> layer5_t;
typedef ap_fixed<8,1> weight5_t;
typedef ap_fixed<8,1> bias5_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer7_t;
typedef ap_fixed<18,8> conv2_relu_table_t;
typedef nnet::array<ap_ufixed<8,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer20_t;
typedef ap_fixed<32,14> conv3_accum_t;
typedef nnet::array<ap_fixed<32,14>, 8*1> layer8_t;
typedef ap_fixed<8,1> weight8_t;
typedef ap_fixed<8,1> bias8_t;
typedef nnet::array<ap_ufixed<8,3,AP_RND_CONV,AP_SAT,0>, 8*1> layer10_t;
typedef ap_fixed<18,8> conv3_relu_table_t;
typedef nnet::array<ap_ufixed<8,3,AP_RND_CONV,AP_SAT,0>, 8*1> layer21_t;
typedef ap_fixed<32,14> conv_out_accum_t;
typedef nnet::array<ap_fixed<32,14>, 2*1> layer11_t;
typedef ap_fixed<8,1> weight11_t;
typedef ap_fixed<8,1> bias11_t;
typedef ap_fixed<32,14> pick_iq_accum_t;
typedef nnet::array<ap_fixed<32,14>, 2*1> layer13_t;
typedef ap_fixed<2,2> weight13_t;
typedef ap_uint<1> bias13_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer14_t;
typedef ap_fixed<18,8> pick_iq_linear_table_t;
typedef nnet::array<ap_fixed<10,3,AP_RND_CONV,AP_SAT,0>, 2*1> layer15_t;
typedef ap_fixed<18,8> conv_out_q_table_t;
typedef nnet::array<ap_fixed<13,5>, 2*1> result_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
