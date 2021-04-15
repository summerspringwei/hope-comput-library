#ifndef ARM_COMPUTE_GRAPH_TYPES_UTILS_HPP
#define ARM_COMPUTE_GRAPH_TYPES_UTILS_HPP

#include "arm_compute/graph/Types.h"

namespace arm_compute
{
namespace graph
{

static std::map<Target, std::string> target_to_str_map = {
    {Target::UNSPECIFIED, "UNSPECIFIED"},
    {Target::NEON, "CPU"},
    {Target::CL, "OpenCL"},
    {Target::GC, "GC"}
};

std::string get_target_string(Target target){
    if(target_to_str_map.find(target) == target_to_str_map.end()){
        return "UNKNOWN_TARGET";
    }else{
        return target_to_str_map[target];
    }
}


static std::map<NodeType, std::string> node_type_to_str_map = {
{NodeType::ActivationLayer, "ActivationLayer" },
{NodeType::ArgMinMaxLayer, "ArgMinMaxLayer" },
{NodeType::BatchNormalizationLayer, "BatchNormalizationLayer" },
{NodeType::BoundingBoxTransformLayer, "BoundingBoxTransformLayer" },
{NodeType::ChannelShuffleLayer, "ChannelShuffleLayer" },
{NodeType::ConcatenateLayer, "ConcatenateLayer" },
{NodeType::ConvolutionLayer, "ConvolutionLayer" },
{NodeType::DeconvolutionLayer, "DeconvolutionLayer" },
{NodeType::DepthToSpaceLayer, "DepthToSpaceLayer" },
{NodeType::DepthwiseConvolutionLayer, "DepthwiseConvolutionLayer" },
{NodeType::DequantizationLayer, "DequantizationLayer" },
{NodeType::DetectionOutputLayer, "DetectionOutputLayer" },
{NodeType::DetectionPostProcessLayer, "DetectionPostProcessLayer" },
{NodeType::EltwiseLayer, "EltwiseLayer" },
{NodeType::FlattenLayer, "FlattenLayer" },
{NodeType::FullyConnectedLayer, "FullyConnectedLayer" },
{NodeType::FusedConvolutionBatchNormalizationLayer, "FusedConvolutionBatchNormalizationLayer" },
{NodeType::FusedDepthwiseConvolutionBatchNormalizationLayer, "FusedDepthwiseConvolutionBatchNormalizationLayer" },
{NodeType::GenerateProposalsLayer, "GenerateProposalsLayer" },
{NodeType::L2NormalizeLayer, "L2NormalizeLayer" },
{NodeType::NormalizationLayer, "NormalizationLayer" },
{NodeType::NormalizePlanarYUVLayer, "NormalizePlanarYUVLayer" },
{NodeType::PadLayer, "PadLayer" },
{NodeType::PermuteLayer, "PermuteLayer" },
{NodeType::PoolingLayer, "PoolingLayer" },
{NodeType::PReluLayer, "PReluLayer" },
{NodeType::PrintLayer, "PrintLayer" },
{NodeType::PriorBoxLayer, "PriorBoxLayer" },
{NodeType::QuantizationLayer, "QuantizationLayer" },
{NodeType::ReductionOperationLayer, "ReductionOperationLayer" },
{NodeType::ReorgLayer, "ReorgLayer" },
{NodeType::ReshapeLayer, "ReshapeLayer" },
{NodeType::ResizeLayer, "ResizeLayer" },
{NodeType::ROIAlignLayer, "ROIAlignLayer" },
{NodeType::SoftmaxLayer, "SoftmaxLayer" },
{NodeType::SliceLayer, "SliceLayer" },
{NodeType::SplitLayer, "SplitLayer" },
{NodeType::StackLayer, "StackLayer" },
{NodeType::StridedSliceLayer, "StridedSliceLayer" },
{NodeType::UpsampleLayer, "UpsampleLayer" },
{NodeType::UnaryEltwiseLayer, "UnaryEltwiseLayer" },
{NodeType::Input, "Input" },
{NodeType::Output, "Output" },
{NodeType::Const, "Const" },
{NodeType::Dummy, "Dummy" },
};


std::string get_node_type_string(NodeType node_type){
    if(node_type_to_str_map.find(node_type) == node_type_to_str_map.end()) {
        return "UNKNOWN_NODE_TYPE";
    } else {
        return node_type_to_str_map[node_type];
    }
}



} // end of namespace graph
} // end of namespace arm_compute
#endif