/*
 * Copyright (c) 2017-2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "graph_nasnet_utils.h"

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cmath>

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

struct Net_And_CellOutputs {
    std::shared_ptr<SubStream> net;
    std::vector<std::shared_ptr<SubStream>> cell_output;
};

template<typename T>
bool vector_find(std::vector<T>& vec, T val){
    for(auto num: vec){
        if(val == num){
            return true;
        }
    }
    return false;
}

Net_And_CellOutputs _imagenet_stem(std::shared_ptr<SubStream> graph, HParams hparams,
                                NasNetABaseCell stem, std::string prefix){
    int num_stem_cells = 2;
    int num_stem_filters = 32 * hparams.stem_multiplier;
    
    *graph << ConvolutionLayer(
            3U, 3U, num_stem_filters,
            std::make_unique<DummyAccessor>(), 
            std::make_unique<DummyAccessor>(), 
            PadStrideInfo(2, 2, 0, 0)).set_name(prefix +"_conv1");
    // Run the reduction cells
    std::vector<std::shared_ptr<SubStream>> cell_outputs = {nullptr, graph};
    print_tensor_shape(get_tail_shape(graph));//point 1
    auto filter_scaling = 1.0 / std::pow(hparams.filter_scaling_rate, num_stem_cells);
    for(int cell_num=0; cell_num<num_stem_cells;cell_num++){
        graph = stem(graph, prefix+std::to_string(cell_num), 
            filter_scaling, 2, cell_outputs[cell_outputs.size()-2], cell_num, nullptr);
        // TODO(Chunwei xia) verify it
        // cell_outputs.push_back(graph);
        cell_outputs.push_back(std::shared_ptr<SubStream>(new SubStream(*graph)));
        filter_scaling = filter_scaling * hparams.filter_scaling_rate;
    }
    return Net_And_CellOutputs{graph, cell_outputs};
}

/** Example demonstrating how to implement InceptionV3's network using the Compute Library's graph API */
class NasnetLargeExample : public Example
{
public:
    NasnetLargeExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "NasnetLarge")
    {
        net_params = HParams{
            3,   // int stem_multiplier;
            18,  // int num_cells;
            2,   // int filter_scaling_rate;
            168, // int num_conv_filters;
            0.7, // float drop_path_keep_prob;
            1,   // int use_aux_head;
            2,   // int num_reduction_layers;
            DataLayout::NCHW, // DataLayout data_format;
            1    // int skip_reduction_layer_input;                    
        };
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = std::make_unique<TFPreproccessor>();

        // Create input descriptor
        const auto        operation_layout = common_params.data_layout;
        const TensorShape tensor_shape     = permute_shape(TensorShape(331U, 331U ,3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;
        ARM_COMPUTE_UNUSED(weights_layout);
        // Calculate the total number of cells in the network
        // Add 2 for the reduction cell and add additional 2 for the stem cells
        int total_num_cells = net_params.num_cells + 2 + 2;
        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false));
        
        NasNetANormalCell normal_cell(net_params.num_conv_filters, net_params.drop_path_keep_prob,
            total_num_cells, 1, false);
        NasNetAReductionCell reduction_cell(net_params.num_conv_filters, net_params.drop_path_keep_prob,
            total_num_cells, 1, false);

        std::shared_ptr<SubStream> net(new SubStream(graph));
        // NasNetAReductionCell stem_cell(net_params.num_conv_filters, net_params.drop_path_keep_prob,
        //     total_num_cells, 1, false);
        auto net_and_cell_outputs = _imagenet_stem(net, net_params, reduction_cell, "imagenet_stem");
        
        auto reduction_indices = calc_reduction_layers(net_params.num_cells, net_params.num_reduction_layers);
        std::vector<int> aux_head_cell_idxes;
        if (reduction_indices.size() >= 2){
            aux_head_cell_idxes.push_back(reduction_indices[1] - 1);
        }
        
        // Run the cells
        float filter_scaling = 1.0;
        int true_cell_num = 2;
        
        net = (net_and_cell_outputs.net);
        for(int cell_num=0; cell_num<net_params.num_cells; ++cell_num){
            int stride = 1;
            std::shared_ptr<SubStream> prev_layer;
            if(net_params.skip_reduction_layer_input){
                prev_layer = net_and_cell_outputs.cell_output[net_and_cell_outputs.cell_output.size()-2];
            }
            
            if(vector_find<int>(reduction_indices, cell_num)){
                filter_scaling *= net_params.filter_scaling_rate;
            }
            
            net = reduction_cell(net, "reduction_cell_"+std::to_string(reduction_indices[cell_num]),
                filter_scaling, 2, net_and_cell_outputs.cell_output[net_and_cell_outputs.cell_output.size()-2],
                true_cell_num, nullptr);
            // if add_and_check_endpoint(
            //     'Reduction_Cell_{}'.format(reduction_indices.index(cell_num)), net):
            //     return net, end_points
            true_cell_num += 1;
            // net_and_cell_outputs.cell_output.push_back(net);
            net_and_cell_outputs.cell_output.push_back(std::shared_ptr<SubStream>(new SubStream(*net)));
            if(!net_params.skip_reduction_layer_input){
                prev_layer = net_and_cell_outputs.cell_output[net_and_cell_outputs.cell_output.size()-2];
            }
            net = normal_cell(net, "cell_"+std::to_string(cell_num),
                filter_scaling, stride, prev_layer, true_cell_num, nullptr);
            // if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
            //     return net, end_points
            true_cell_num += 1;
            // Omit the operation only needed for training
            // net_and_cell_outputs.cell_output.push_back(net);
            net_and_cell_outputs.cell_output.push_back(std::shared_ptr<SubStream>(new SubStream(*net)));
        }
        graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        graph << FullyConnectedLayer(1001, std::make_unique<DummyAccessor>(), 
            std::make_unique<DummyAccessor>());
        graph << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads      = common_params.threads;
        config.use_tuner        = common_params.enable_tuner;
        config.tuner_mode       = common_params.tuner_mode;
        config.tuner_file       = common_params.tuner_file;
        config.mlgo_file        = common_params.mlgo_file;
        config.convert_to_uint8 = (common_params.data_type == DataType::QASYMM8);
        config.execution_type   = common_params.execution_type;
        config.device_map_file  = common_params.device_map_file;

        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        // graph.run();
        graph.run(common_params.num_runs);
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
    HParams            net_params;

private:
    

};

/** Main program for Inception V3
 *
 * Model is based on:
 *      https://arxiv.org/abs/1512.00567
 *      "Rethinking the Inception Architecture for Computer Vision"
 *      Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
 *
 * Provenance: download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<NasnetLargeExample>(argc, argv);
}
