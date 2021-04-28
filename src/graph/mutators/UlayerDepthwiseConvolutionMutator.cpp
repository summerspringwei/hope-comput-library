
#include "arm_compute/graph/mutators/UlayerDepthwiseConvolutionMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphBuilder.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/nodes/Nodes.h"

#include "support/Cast.h"

#include "support/StringSupport.h"

#include <set>

namespace arm_compute
{
namespace graph
{
namespace
{
/**
 * Note:For now, We only split the depthwise to 2 
*/
NodeID create_ulayer_depthwise_convolution(Graph &g, const NodeParams &params, NodeIdxPair input, NodeID weights, NodeID bias,
                                PadStrideInfo conv_info, float ratio){
    bool has_bias = (bias != EmptyNodeID);
    const int num_splits = 2;
    // Split input
    const TensorDescriptor input_tensor_desc = get_tensor_descriptor(g, g.node(input.node_id)->outputs()[0]);
    const unsigned int     input_idx         = get_dimension_idx(input_tensor_desc.layout, DataLayoutDimension::CHANNEL);
    const unsigned int     input_channel     = input_tensor_desc.shape[input_idx];
    // Compute splited channel number
    int device1_channel_num = input_channel * ratio;
    int device2_channel_num = input_channel - device1_channel_num;
    NodeID                input_split        = GraphBuilder::add_split_node(g, params, input, num_splits, input_idx, {device1_channel_num, device2_channel_num});
    
        // Split weights
    const TensorDescriptor weights_tensor_desc = get_tensor_descriptor(g, g.node(weights)->outputs()[0]);
    const unsigned int     batch_idx           = get_dimension_idx(weights_tensor_desc.layout, DataLayoutDimension::BATCHES);
    NodeID                 weights_split       = GraphBuilder::add_split_node(g, params, { weights, 0 }, num_splits, batch_idx, {device1_channel_num, device2_channel_num});
    // Split bias
    NodeID bias_split = EmptyNodeID;
    if(has_bias)
    {
        // Split bias
        bias_split = GraphBuilder::add_split_node(g, params, { bias, 0 }, num_splits, 0, {device1_channel_num, device2_channel_num});
    }
    std::vector<NodeIdxPair> depthwise_convolution_outputs;
    for(int i=0 ;i<num_splits; ++i){
        NodeParams depth_conv_params = params;
        NodeID depth_conv_id = g.add_node<DepthwiseConvolutionLayerNode>(conv_info);
        g.add_connection(input_split, i, depth_conv_id, 0);
        g.add_connection(weights_split, i, depth_conv_id, 1);
        if(has_bias)
        {
            g.add_connection(bias_split, i, depth_conv_id, 2);
        }
        // Set node parameters
        auto node = g.node(depth_conv_id);
        ARM_COMPUTE_ERROR_ON(node==nullptr);
        node->set_common_node_parameters(depth_conv_params);
        
        if(i==0) {
            depth_conv_params.name.append("_cpu");
            depth_conv_params.target = Target::NEON;
            node->set_assigned_target(Target::NEON);
        } else {
            depth_conv_params.name.append("_gpu");
            depth_conv_params.target = Target::CL;
            node->set_assigned_target(Target::CL);
        }
        // We omit fused activation

        depthwise_convolution_outputs.push_back({depth_conv_id, 0});
    }
    return GraphBuilder::add_concatenate_node(g, params, depthwise_convolution_outputs, DataLayoutDimension::CHANNEL);
}
}// End of anonymous namespace
const char *UlayerDepthwiseConvolutionMutator::name()
{
    return "UlayerDepthwiseConvolutionMutator";
}

IGraphMutator::MutationType UlayerDepthwiseConvolutionMutator::type() const
{
    return IGraphMutator::MutationType::Backend;
}

void UlayerDepthwiseConvolutionMutator::mutate(Graph& g){
    // Early exit if no Convolution layers exist in graph
    if(g.nodes(NodeType::DepthwiseConvolutionLayer).empty())
    {
        return;
    }
    // Total nodes
    size_t total_nodes = g.nodes().size();
    // Iterate over convolution nodes
    for(unsigned int i = 0; i < total_nodes; ++i)
    {
        INode *node = g.node(i);
        if(node!=nullptr && node->type()==NodeType::DepthwiseConvolutionLayer){
            // Validate node
            backends::IDeviceBackend &backend = backends::BackendRegistry::get().get_backend(node->assigned_target());
            Status                    status  = backend.validate_node(*node);

            if(!bool(status)){
                // Down-cast node
                auto *depth_conv_node = arm_compute::utils::cast::polymorphic_cast<DepthwiseConvolutionLayerNode *>(node);
                

                // Get internal convolution info
                const PadStrideInfo       conv_info       = depth_conv_node->convolution_info();
                auto params = depth_conv_node->common_node_params();
                // Extract node ids
                ARM_COMPUTE_ERROR_ON(depth_conv_node->input_edge(0) == nullptr || depth_conv_node->input_edge(1) == nullptr);
                const NodeID input_id   = depth_conv_node->input_edge(0)->producer()->id();
                const NodeID weights_id = depth_conv_node->input_edge(1)->producer()->id();
                const NodeID bias_id    = (depth_conv_node->input_edge(2) != nullptr) ? depth_conv_node->input_edge(2)->producer()->id() : EmptyNodeID;

                // Get driving nodes
                std::vector<NodeIdxPair> driving_nodes = get_driving_nodes(*node);
                // Extract activation node accessor if any
                auto node_accessor = depth_conv_node->output(0)->extract_accessor();

                NodeID ulayer_depth_conv_id = create_ulayer_depthwise_convolution(g, params, {input_id, 0}, weights_id, bias_id,
                    conv_info, _ratio);
                
                g.remove_node(node->id());

                // Update output nodes edges
                for(auto &driving_node : driving_nodes)
                {
                    g.add_connection(ulayer_depth_conv_id, 0, driving_node.node_id, driving_node.index);
                }
                // // Current max tensor and node id
                // TensorID latest_tid = g.tensors().size();
                // NodeID   latest_nid = g.nodes().size();

                // Update accessor to batch normalization node
                g.node(ulayer_depth_conv_id)->output(0)->set_accessor(std::move(node_accessor));

                // we have config node target in create_ulayer_depthwise_convolution
                // We will configure the tensor after the pass manager

            }
        }
    }
}



}
}