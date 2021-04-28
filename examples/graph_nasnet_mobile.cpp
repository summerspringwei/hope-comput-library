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


/** Example demonstrating how to implement InceptionV3's network using the Compute Library's graph API */
class NasnetMobileExample : public Example
{
public:
    NasnetMobileExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "NasnetMobile")
    {
        net_params = HParams{
            1,   // int stem_multiplier;
            12,  // int num_cells;
            2,   // int filter_scaling_rate;
            44, // int num_conv_filters;
            1.0, // float drop_path_keep_prob;
            1,   // int use_aux_head;
            2,   // int num_reduction_layers;
            DataLayout::NCHW, // DataLayout data_format;
            0    // int skip_reduction_layer_input;                    
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
        const TensorShape tensor_shape     = permute_shape(TensorShape(224U, 224U ,3U, 1U), DataLayout::NCHW, operation_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(operation_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;
        ARM_COMPUTE_UNUSED(weights_layout);
        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false));
        build_nasnet_base(graph, net_params);

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

/** Main program for NasNet Mobile
 *
 * Model is based on:
 *      https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet
 * We used the default configuration for NasNet-Mobile
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<NasnetMobileExample>(argc, argv);
}
