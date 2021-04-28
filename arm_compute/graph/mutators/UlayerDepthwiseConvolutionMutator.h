

#ifndef ARM_COMPUTE_GRAPH_ULAYER_DEPTHWISE_CONVOLUTION_MUTATOR_H
#define ARM_COMPUTE_GRAPH_ULAYER_DEPTHWISE_CONVOLUTION_MUTATOR_H

#include "arm_compute/graph/IGraphMutator.h"

namespace arm_compute
{
namespace graph
{
/** Mutation pass to implement Ulayer for Depthwise Convolution
 * See paper "μLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization"
 * for more details
 **/
class UlayerDepthwiseConvolutionMutator final: public IGraphMutator
{
public:
    /**
     * @param [in] ratio: the ratio of workload that is assigned to cpu
    */
    UlayerDepthwiseConvolutionMutator(float ratio): _ratio(ratio){
    }
    virtual void mutate(Graph& g) override;
    MutationType type() const override;
    const char *name() override;

private:
    float _ratio;
};

}
}

#endif