#include <iostream>

#include "chest.h"
#include "parameters.h"


void chest(
    hls::stream<input_t> &h_ls,
    hls::stream<result_t> &layer16_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=h_ls,layer16_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<weight2_t, 432>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 2304>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 1152>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 8>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight11_t, 144>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 2>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight13_t, 6>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 2>(b13, "b13.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<input_t> layer17_cpy1("layer17_cpy1");
    #pragma HLS STREAM variable=layer17_cpy1 depth=1008

    hls::stream<input_t> layer17_cpy2("layer17_cpy2");
    #pragma HLS STREAM variable=layer17_cpy2 depth=1008

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=1184

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1008

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1008

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=1184

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1008

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=1008

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=1184

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1008

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1008

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=1184

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=1008

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1008

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=1008

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=1008

    nnet::clone_stream<input_t, input_t, 3024>(h_ls, layer17_cpy1, layer17_cpy2); // clone_h_ls

    nnet::zeropad2d_cl<input_t, layer18_t, config18>(layer17_cpy1, layer18_out); // zp2d_conv1

    nnet::conv_2d_cl<layer18_t, layer2_t, config2>(layer18_out, layer2_out, w2, b2); // conv1

    nnet::relu<layer2_t, layer4_t, relu_config4>(layer2_out, layer4_out); // conv1_relu

    nnet::zeropad2d_cl<layer4_t, layer19_t, config19>(layer4_out, layer19_out); // zp2d_conv2

    nnet::conv_2d_cl<layer19_t, layer5_t, config5>(layer19_out, layer5_out, w5, b5); // conv2

    nnet::relu<layer5_t, layer7_t, relu_config7>(layer5_out, layer7_out); // conv2_relu

    nnet::zeropad2d_cl<layer7_t, layer20_t, config20>(layer7_out, layer20_out); // zp2d_conv3

    nnet::conv_2d_cl<layer20_t, layer8_t, config8>(layer20_out, layer8_out, w8, b8); // conv3

    nnet::relu<layer8_t, layer10_t, relu_config10>(layer8_out, layer10_out); // conv3_relu

    nnet::zeropad2d_cl<layer10_t, layer21_t, config21>(layer10_out, layer21_out); // zp2d_conv_out

    nnet::conv_2d_cl<layer21_t, layer11_t, config11>(layer21_out, layer11_out, w11, b11); // conv_out

    nnet::pointwise_conv_2d_cl<input_t, layer13_t, config22>(layer17_cpy2, layer13_out, w13, b13); // pick_iq

    nnet::linear<layer13_t, layer14_t, linear_config14>(layer13_out, layer14_out); // pick_iq_linear

    nnet::linear<layer11_t, layer15_t, linear_config15>(layer11_out, layer15_out); // conv_out_q

    nnet::add<layer14_t, layer15_t, result_t, config16>(layer14_out, layer15_out, layer16_out); // residual

}

