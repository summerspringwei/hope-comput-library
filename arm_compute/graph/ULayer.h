
#ifndef ULAYER_H
#define ULAYER_H

#include "arm_compute/graph/frontend/ILayer.h"
#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph/frontend/IStream.h"
#include "arm_compute/graph.h"
#include "utils/GraphUtils.h"

namespace arm_compute
{
namespace graph
{
namespace frontend
{

ConcatLayer UConvolutionLayer(
                     Stream &               graph,
                     unsigned int           conv_width,
                     unsigned int           conv_height,
                     unsigned int           ofm,
                     ITensorAccessorUPtr    weights,
                     ITensorAccessorUPtr    bias,
                     PadStrideInfo          conv_info,
                     unsigned int           num_groups         = 1,
                     float                  ratio = 0,
                     std::string            name = "UConv",
                     const QuantizationInfo weights_quant_info = QuantizationInfo(),
                     const QuantizationInfo out_quant_info     = QuantizationInfo()){
    int device_0_ofm = ofm * ratio;
    int device_1_ofm = ofm - device_0_ofm;
    SubStream c_0(graph);
    SubStream c_1(graph);
    c_0 << ConvolutionLayer(conv_width, conv_height, device_0_ofm, 
            std::move(weights), 
            std::move(bias), conv_info, num_groups, weights_quant_info, out_quant_info)
            .set_name(name+"_cpu");
    c_1 << ConvolutionLayer(conv_width, conv_height, device_1_ofm, 
            std::make_unique<arm_compute::graph_utils::DummyAccessor>(), 
            std::make_unique<arm_compute::graph_utils::DummyAccessor>(), conv_info, num_groups, weights_quant_info, out_quant_info)
            .set_name(name+"_gpu");
    ConcatLayer concat(std::move(c_0), std::move(c_1));
    concat.set_name(name+"_concat");
    return concat;
}

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

// void USeperableConvolutionLayer(
//                      Stream &               graph,
//                      unsigned int           conv_width,
//                      unsigned int           conv_height,
//                      unsigned int           ofm,
//                      ITensorAccessorUPtr    weights,
//                      ITensorAccessorUPtr    bias,
//                      PadStrideInfo          conv_info,
//                      unsigned int           num_groups         = 1,
//                      float                  ratio = 0,
//                      std::string            name = "UConv",
//                      const QuantizationInfo weights_quant_info = QuantizationInfo(),
//                      const QuantizationInfo out_quant_info     = QuantizationInfo()){
//     std::shared_ptr<SubStream> c_0(new SubStream(graph));
//     std::shared_ptr<SubStream> c_1(new SubStream(graph));
    
//     auto filter_num = get_tail_shape(c_0)[2];
//     int device_0_input_filter_num = filter_num * ratio;
//     int device_1_input_filter = filter_num - device_0_input_filter_num;
//     auto input_tid = graph.graph().node(graph.tail_node())->outputs()[0];
//     auto input_node_id = graph.tail_node();
    
//     auto params = graph.graph().node(input_node_id)->common_node_params();
//     const int num_splits = 2;
//     const TensorDescriptor input_tensor_desc = get_tensor_descriptor(graph.graph(), input_tid);
//     const unsigned int     input_idx         = get_dimension_idx(input_tensor_desc.layout, DataLayoutDimension::CHANNEL);
//     NodeID                 input_split = GraphBuilder::add_split_node(graph.graph(), 
//                                             params,
//                                             {input_node_id, 0}, num_splits, input_idx);
//     std::vector<NodeIdxPair> depth_conv_node_idxes;
//     for(int i=0; i<num_splits; ++i){
//         NodeParams depth_params;
//         if(i==0){
//             depth_params.name = params.name+"_cpu";
//             depth_params.target = Target::NEON;
//         }else{
//             depth_params.name = params.name+"_gpu";
//             depth_params.target = Target::CL;
//         }
        
//         auto node_id = GraphBuilder::add_depthwise_convolution_node(graph.graph(), depth_params,
//             {input_split, 0}, Size2D(conv_width, conv_height), conv_info, 1,
//             graph.hints().depthwise_convolution_method_hint,
//             std::make_unique<DummyAccessor>(), 
//             std::make_unique<DummyAccessor>(), weights_quant_info, out_quant_info);
//         depth_conv_node_idxes.push_back({node_id, 0});
//     }
    
//     GraphBuilder::add_concatenate_node(graph.graph(), params, depth_conv_node_idxes, DataLayoutDimension::CHANNEL);
// }

}
}
}

#endif