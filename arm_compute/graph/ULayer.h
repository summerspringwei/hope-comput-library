
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

}
}
}

#endif