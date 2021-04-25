#ifndef GRAPH_NASNET_UTILS_H
#define GRAPH_NASNET_UTILS_H

#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

struct HParams
{
    /* data */
    int stem_multiplier;
    int num_cells;
    int filter_scaling_rate;
    int num_conv_filters;
    float drop_path_keep_prob;
    int use_aux_head;
    int num_reduction_layers;
    DataLayout data_format;
    int skip_reduction_layer_input;
};

// Build test pass
/**Figure out what layers should have reductions */
std::vector<int> calc_reduction_layers(int num_cells, int num_reduction_layers){
    std::vector<int> reduction_layers;
    for(int pool_num = 1; pool_num < num_reduction_layers+1; ++pool_num){
        float layer_num_f = (float)pool_num / (num_reduction_layers + 1) * num_cells;
        int layer_num_i = (int)layer_num_f;
        reduction_layers.push_back(layer_num_i);
    }
    return reduction_layers;
}

// int _operation_to_num_layers(std::string operation){

// };

// int _operation_to_filter_shape(std::string operation){

// }

// for string delimiter
std::vector<std::string> split (std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

// Build test pass
std::vector<int> _operation_to_info(std::string operation){
    // separable_3x3_4 -> {"separable"}
    auto com = split(operation, "_");
    auto x_pos = com[1].find("x");
    int filter_size = atoi((com[1].substr(0, x_pos)).c_str());
    int num_layers = atoi(com[2].c_str());
    return {num_layers, filter_size};
};

// Build test pass
std::vector<std::string> _operation_to_pooling_info(std::string operation){
    // 'avg_pool_3x3' -> {"avg", 3}
    auto com = split(operation, "_");
    auto window_size_str = split(com[2], "x")[0];
    return {com[0], window_size_str};
}

/**Parse operation and performs the correct polling operation on net
 * 
*/
std::shared_ptr<SubStream> _pooling(std::shared_ptr<SubStream> graph, int stride, 
    std::string operation, bool use_bounded_activation, std::string prefix){
    ARM_COMPUTE_UNUSED(prefix);
    auto pooling_info_str = _operation_to_pooling_info(operation);
    auto pooling_type_str = pooling_info_str[0];
    int pooling_shape = atoi(pooling_info_str[1].c_str());
    if(use_bounded_activation){
        *graph<<ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU, 0, 6));
    }
    if(pooling_type_str=="avg"){
        *graph<<PoolingLayer(PoolingLayerInfo(PoolingType::AVG, pooling_shape, DataLayout::NCHW, 
            PadStrideInfo(stride, stride, pooling_shape / 2, pooling_shape / 2)));// Padding SAME
    }else if(pooling_type_str == "max"){
        *graph<<PoolingLayer(PoolingLayerInfo(PoolingType::MAX, pooling_shape, DataLayout::NCHW, 
            PadStrideInfo(stride, stride, pooling_shape / 2, pooling_shape / 2)));// Padding SAME
    }else{
        ARM_COMPUTE_ERROR(("Unimplemented pooling type" +pooling_type_str).c_str());
    }
    return graph;
}

// TODO(Chunwei Xia) Support NHWC data layout
TensorShape get_tail_shape(std::shared_ptr<SubStream> layer){
    if(layer == nullptr){
        return TensorShape(0,0,0,0);
    }
    auto node = layer->graph().node(layer->tail_node());
    if(node==nullptr || node->num_outputs() < 1){
        return TensorShape(0,0,0,0);
    }
    ARM_COMPUTE_ERROR_ON(node->output(0)->desc().layout != DataLayout::NCHW);
    return node->output(0)->desc().shape;
}

void print_tensor_shape(TensorShape shape){
    printf("[%ld, %ld, %ld, %ld]\n", shape[0], shape[1], shape[2], shape[3]);
}

/**
 * NASNet cell class that is used as a 'layer' in image architechture
 * 
*/
class NasNetABaseCell{

public:
    /**
     * @param [in] num_conv_filters: The number of filters for each convolution operation
     * @param [in] operations: List of operations that are performed for each convolution operation
     * @param [in] used_hiddenstates: Binary array that signals if the hidden state was used within the cell.
     *     this is used to determine what outputs of the cell should be concatednate together.
     * @param [in] hiddenstate_indices: Determines what hiddenstates should be combined together
     *     with the specific operations to create the NASNet cell.
     * 
    */
    NasNetABaseCell(int num_conv_filters, std::vector<std::string> operations, 
                    std::vector<int> used_hiddenstates, std::vector<int> hiddenstate_indices,
                    float drop_path_keep_prob, int total_num_cells,
                    int total_training_steps, bool used_bounded_activation=false)
                    :_num_conv_filters(num_conv_filters), _operations(operations),
                    _used_hiddenstates(used_hiddenstates), _hiddenstate_indices(hiddenstate_indices),
                    _drop_path_keep_prob(drop_path_keep_prob), _total_num_cells(total_num_cells),
                    _total_training_steps(total_training_steps), _used_bounded_activation(used_bounded_activation) {
        ARM_COMPUTE_UNUSED(_drop_path_keep_prob);
        ARM_COMPUTE_UNUSED(_total_num_cells);
        ARM_COMPUTE_UNUSED(_total_training_steps);
    }
    // Code check pass
    // TODO: to see the tensorflow's python implementation of separable conv
    std::shared_ptr<SubStream> _stacked_separable_conv(std::shared_ptr<SubStream> graph, int stride, std::string operation, 
                                        int filter_size, bool use_bounded_activation, std::string prefix){
        ARM_COMPUTE_UNUSED(use_bounded_activation);
        auto op_info = _operation_to_info(operation);
        int num_layers = op_info[0];
        int kernel_size = op_info[1];
        
        std::stringstream ss;
        for(int layer_num = 0; layer_num < num_layers - 1; layer_num++){
            *graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
            ss.clear();
            ss << prefix << "separable_depthwise" << kernel_size << "x" << kernel_size << "_" << (layer_num+1);
            *graph << DepthwiseConvolutionLayer(kernel_size, kernel_size,
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(stride, stride, kernel_size / 2, kernel_size / 2)
            ).set_name(ss.str());
            ss.clear();
            ss << prefix << "separable_pointwise" << kernel_size << "x" << kernel_size << "_" << (layer_num+1);
            *graph << ConvolutionLayer(1U, 1U, filter_size,
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(1, 1, 0, 0)).set_name(ss.str());
            *graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
            stride = 1;
        }
        ss.clear();
        ss << prefix << "separable_depthwise" << kernel_size << "x" << kernel_size << "_" << (num_layers);
        *graph << DepthwiseConvolutionLayer(kernel_size, kernel_size,
            std::make_unique<DummyAccessor>(), 
            std::make_unique<DummyAccessor>(), 
            PadStrideInfo(stride, stride, kernel_size / 2, kernel_size / 2)
        ).set_name(ss.str());
        ss.clear();
        ss << prefix << "separable_pointwise" << kernel_size << "x" << kernel_size << "_" << (num_layers);
        *graph << ConvolutionLayer(1U, 1U, filter_size,
            std::make_unique<DummyAccessor>(), 
            std::make_unique<DummyAccessor>(), 
            PadStrideInfo(1, 1, 0, 0)).set_name(ss.str());
        return graph;
    }

    // Code check pass
    /** Applies the predicted conv operations to net */
    std::shared_ptr<SubStream> _apply_conv_operation(std::shared_ptr<SubStream> graph, std::string operation, int stride, 
                                    bool is_from_original_input, std::shared_ptr<SubStream> current_step, std::string prefix){
        // Dont stride if this is not one of the original hiddent state
        if(stride > 1 && !is_from_original_input){
            stride = 1;
        }
        // Assume NCHW
        int input_filters = get_tail_shape(graph)[1];
        ARM_COMPUTE_ERROR_ON(input_filters==0);
        auto filter_size = this->_filter_size;
        if(operation.find("separable", 0) != std::string::npos){
            _stacked_separable_conv(graph, stride, operation, filter_size, false, prefix);
        }else if(operation.find("none") != std::string::npos){
            if((stride > 1) || (input_filters != filter_size)){
                *graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
                *graph << ConvolutionLayer(1U, 1U, filter_size,
                    std::make_unique<DummyAccessor>(), 
                    std::make_unique<DummyAccessor>(),
                    PadStrideInfo(stride, stride, 0, 0))
                    .set_name(prefix+"1x1");
            }
        }else if(operation.find("pool") != std::string::npos){
            _pooling(graph, stride, operation, _used_bounded_activation, prefix+"_pooling_");
            if(input_filters != filter_size){
                *graph << ConvolutionLayer(1U, 1U, filter_size,
                    std::make_unique<DummyAccessor>(), 
                    std::make_unique<DummyAccessor>(),
                    PadStrideInfo(1, 1, 0, 0))
                    .set_name(prefix + "pool_1x1_conv");
            }
        }else{
            ARM_COMPUTE_ERROR(("Unimplemented operation " + operation).c_str());
        }
        // We does not add drop layer
        ARM_COMPUTE_UNUSED(current_step);
        return graph;
    }
    // Code check pass
    /**Concatenate the unused hiddent states of the cell.*/
    std::shared_ptr<SubStream> _combine_unused_states(std::vector<std::shared_ptr<SubStream>> net, std::string prefix){
        std::vector<int> used_hiddenstates = _used_hiddenstates;
        // We assume tensor in NCHW format
        auto final_height = get_tail_shape(net.back())[2];
        ARM_COMPUTE_ERROR_ON(final_height==0);
        auto final_num_filters = get_tail_shape(net.back())[1];
        ARM_COMPUTE_ERROR_ON(final_num_filters == 0);
        assert(net.size() == used_hiddenstates.size());
        
        for(uint64_t idx = 0; idx < used_hiddenstates.size(); idx++){
            auto used_h = used_hiddenstates[idx];
            auto current_height = get_tail_shape(net[idx])[2];
            ARM_COMPUTE_ERROR_ON(current_height==0);
            auto current_num_filters = get_tail_shape(net[idx])[1];
            ARM_COMPUTE_ERROR_ON(current_num_filters==0);
            
            // Determin if a reduction should be applied to make the number of filters match
            bool should_reduce = (final_num_filters != current_num_filters);
            should_reduce = (final_height != current_height) || should_reduce;
            should_reduce = should_reduce && (!used_h);

            if(should_reduce){
                int stride = (final_height != current_height ? 2: 1);
                std::string name_scope = prefix+"_reducetion_" + std::to_string(idx);
                factorized_reduction((net[idx]), final_num_filters,
                    stride, DataLayout::NCHW, name_scope);
            }
        }
        std::shared_ptr<SubStream> out(net[0]);
        std::vector<std::shared_ptr<SubStream>> states_to_combine;
        for(size_t i=0; i<net.size(); ++i){
            if(!used_hiddenstates[i]){
                states_to_combine.push_back(net[i]);
            }
        }
        switch (states_to_combine.size())
        {
        case 2:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), 
                        std::move(*states_to_combine[1]));
            break;
        case 3:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), std::move(*states_to_combine[1]),
                        std::move(*states_to_combine[2]));
            break;
        case 4:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), std::move(*states_to_combine[1]),
                        std::move(*states_to_combine[2]), std::move(*states_to_combine[3]));
            break;
        case 5:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), std::move(*states_to_combine[1]),
                        std::move(*states_to_combine[2]), std::move(*states_to_combine[3]),
                        std::move(*states_to_combine[4]));
            break;
        case 6:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), std::move(*states_to_combine[1]),
                        std::move(*states_to_combine[2]), std::move(*states_to_combine[3]),
                        std::move(*states_to_combine[4]), std::move(*states_to_combine[5]));
            break;
        case 7:
            *out<<ConcatLayer(std::move(*states_to_combine[0]), std::move(*states_to_combine[1]),
                        std::move(*states_to_combine[2]), std::move(*states_to_combine[3]),
                        std::move(*states_to_combine[4]), std::move(*states_to_combine[5]),
                        std::move(*states_to_combine[6]));
        default:
            ARM_COMPUTE_ERROR("Size of states_to_combine out of range\n");
            break;
        }
        
        return out;
    }

    // Code check pass
    std::shared_ptr<SubStream> operator()(std::shared_ptr<SubStream> graph, std::string prefix, float filter_scaling=1.0, int stride=1,
                            std::shared_ptr<SubStream> prev_layer=nullptr, int cell_num=-1, std::shared_ptr<SubStream> current_step=nullptr){
        _cell_num = cell_num;
        _filter_scaling = filter_scaling;
        _filter_size = (int)(_num_conv_filters * filter_scaling);
        printf("Before cell base net:");
        print_tensor_shape(get_tail_shape(graph));
        if(prev_layer != nullptr){
            printf("Before cell base prev_layer:");
            print_tensor_shape(get_tail_shape(prev_layer));
        }
        auto net = _cell_base(graph, prev_layer, prefix+"_cell_base_");
        
        for(int i=0; i<10; i+=2){
            std::string name_scope = prefix + "comb_iter_"+std::to_string(i);
            int left_hiddenstate_idx = _hiddenstate_indices[i];
            int right_hiddenstate_idx = _hiddenstate_indices[i+1];
            bool original_input_left = left_hiddenstate_idx < 2? true: false;
            bool original_input_right = right_hiddenstate_idx < 2? true: false;
            auto h1 = net[left_hiddenstate_idx];
            auto h2 = net[right_hiddenstate_idx];
            
            std::string operation_left = _operations[i];
            std::string operation_right = _operations[i+1];
            printf("before cell_num:%d i:%d\n", cell_num, i);
            print_tensor_shape(get_tail_shape(h1));
            print_tensor_shape(get_tail_shape(h2));
            h1 = _apply_conv_operation(h1, operation_left, stride, original_input_left, current_step, name_scope+"left");
            h2 = _apply_conv_operation(h2, operation_right, stride, original_input_right, current_step, name_scope+"right");
            printf("after cell_num:%d i:%d\n", cell_num, i);
            print_tensor_shape(get_tail_shape(h1));
            print_tensor_shape(get_tail_shape(h2));
            // Combine hidden states using 'add'
            // std::shared_ptr<SubStream> h(h1);
            std::shared_ptr<SubStream> h(new SubStream(*h1));
            *h << EltwiseLayer(std::move(*h1), std::move(*h2), EltwiseOperation::Add);
            net.push_back(h);
        }
        return _combine_unused_states(net, prefix + "cell_output");
    }
    
    // Code check pass.
    std::vector<std::shared_ptr<SubStream>> _cell_base(std::shared_ptr<SubStream> graph, 
        std::shared_ptr<SubStream> prev_layer, std::string prefix) {
        int num_filters = this->_filter_size;
        
        // Check to be sure prev layer stuff is setup correctly
        prev_layer = _reduce_prev_layer(prev_layer, graph, prefix+"_reduce_prev_layer_");
        printf("After _reduce_prev_layer: ");
        print_tensor_shape(get_tail_shape(prev_layer));
        // We directly use relu here
        *graph << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        *graph << ConvolutionLayer(1U, 1U, num_filters,
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(1, 1, 0, 0))
                .set_name(prefix+"1x1");
        return {graph, prev_layer};
    }
    
    // Code check pass
    /**
     * Matches dimension of prev_layer to the current layer
     * including filter_size and filter_number
    */
    std::shared_ptr<SubStream> _reduce_prev_layer(std::shared_ptr<SubStream> prev_layer, 
        std::shared_ptr<SubStream> current_layer, std::string prefix){
        if(prev_layer == nullptr){
            return current_layer;
        }
        uint64_t curr_num_filters = this->_filter_size;
        // Assert NCHW
        auto prev_num_filters = get_tail_shape(prev_layer)[1];
        ARM_COMPUTE_ERROR_ON(prev_num_filters == 0);
        auto current_filter_shape = get_tail_shape(current_layer)[2];
        ARM_COMPUTE_ERROR_ON(current_filter_shape == 0);
        auto prev_filter_shape = get_tail_shape(prev_layer)[2];

        if(current_filter_shape != prev_filter_shape){
            (*prev_layer) << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
            factorized_reduction(prev_layer, curr_num_filters, 2, DataLayout::NCHW, prefix + "_prev_1x1_");
        }else if(curr_num_filters != prev_num_filters){
            (*prev_layer) << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
            (*prev_layer) << ConvolutionLayer(1, 1, curr_num_filters,
                            std::make_unique<DummyAccessor>(), 
                            std::make_unique<DummyAccessor>(), 
                            PadStrideInfo(1, 1, 0, 0));
        }
        return prev_layer;
    }
    // Code check pass
    /** Reduces the shape of net without information loss due to striding */
    std::shared_ptr<SubStream> factorized_reduction(std::shared_ptr<SubStream> graph, int output_filters, int stride, DataLayout data_format, std::string prefix){
        ARM_COMPUTE_ERROR_ON(data_format != DataLayout::NCHW);
        if(stride == 1){
            *graph<< ConvolutionLayer(1, 1, output_filters, 
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(1, 1, 0, 0))
                .set_name(prefix + "path_conv");
            return graph;
        }
        // Skip path 1
        SubStream path1(*graph);
        path1 << PoolingLayer(
                    PoolingLayerInfo(
                        PoolingType::AVG, Size2D(1,1), 
                        DataLayout::NCHW, PadStrideInfo(stride, stride, 0, 0)))
                        .set_name(prefix+"path1_pooling");
        path1 << ConvolutionLayer(1U, 1U, output_filters / 2,
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(1, 1, 0, 0))
                .set_name(prefix+"path1_conv");
        // Skip path 2
        ARM_COMPUTE_ERROR_ON(data_format != DataLayout::NCHW);
        SubStream path2(*graph);
        path2 << PadLayer({{0, 0}, {0, 0}, {0, 1}, {0, 1}});
        path2 << PoolingLayer(
                    PoolingLayerInfo(
                        PoolingType::AVG, Size2D(1,1), 
                        DataLayout::NCHW, PadStrideInfo(stride, stride, 0, 0)))
                        .set_name(prefix+"path2_pooling");
        // If odd number of filters, add an additional one to the second path
        int final_filter_size = (output_filters / 2) + (output_filters % 2);
        path2 << ConvolutionLayer(1U, 1U, final_filter_size,
                std::make_unique<DummyAccessor>(), 
                std::make_unique<DummyAccessor>(), 
                PadStrideInfo(1, 1, 0, 0))
                .set_name(prefix+"path2_conv");
        // Concat
        *graph << ConcatLayer(std::move(path1), std::move(path2)).set_name(prefix + "concat");
        return graph;
    }

private:
    int _num_conv_filters;
    std::vector<std::string> _operations;
    std::vector<int> _used_hiddenstates;
    std::vector<int> _hiddenstate_indices;
    float _drop_path_keep_prob;
    int _total_num_cells;
    int _total_training_steps;
    bool _used_bounded_activation=false;
    int _cell_num;
    float _filter_scaling;
    int _filter_size;
};


class NasNetANormalCell: public NasNetABaseCell{
public:
    NasNetANormalCell(int num_conv_filters, float drop_path_keep_prob, int total_num_cells,
        int total_training_steps, bool use_bound_activation=false) : 
        NasNetABaseCell(num_conv_filters, {
                "separable_5x5_2",
                "separable_3x3_2",
                "separable_5x5_2",
                "separable_3x3_2",
                "avg_pool_3x3",
                "none",
                "avg_pool_3x3",
                "avg_pool_3x3",
                "separable_3x3_2",
                "none"
            }, {1, 0, 0, 0, 0, 0, 0},
                {0, 1, 1, 1, 0, 1, 1, 1, 0, 0}, drop_path_keep_prob, total_num_cells, 
                total_training_steps, use_bound_activation){

    }
};

class NasNetAReductionCell: public NasNetABaseCell {
public:
    NasNetAReductionCell(int num_conv_filters, float drop_path_keep_prob, int total_num_cells,
        int total_training_steps, bool use_bound_activation=false): 
        NasNetABaseCell(num_conv_filters, {
            "separable_5x5_2",
            "separable_7x7_2",
            "max_pool_3x3",
            "separable_7x7_2",
            "avg_pool_3x3",
            "separable_5x5_2",
            "none",
            "avg_pool_3x3",
            "separable_3x3_2",
            "max_pool_3x3"
        }, {1, 1, 1, 0, 0, 0, 0},
                {0, 1, 0, 1, 0, 1, 3, 2, 2, 0}, drop_path_keep_prob, total_num_cells, 
                total_training_steps, use_bound_activation){
        
    }
};


#endif