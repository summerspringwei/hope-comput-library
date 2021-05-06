/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#include "arm_compute/graph/GraphManager.h"

#include <mutex>

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/algorithms/TopologicalSort.h"

namespace arm_compute
{
namespace graph
{

extern std::string get_target_string(Target);

GraphManager::GraphManager()
    : _workloads()
{
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }
    auto execution_type  = ctx.config().execution_type;
    std::shared_ptr<std::map<std::string, Target>> device_map_ptr = nullptr;
    if(execution_type != ExecutionType::EXECUTION_TYPE_DEFAULT){
        // Get device placement
        device_map_ptr = std::move(detail::read_device_map((const char*)ctx.config().device_map_file.c_str()));
        if(device_map_ptr == nullptr){
            printf("Device placement is null\n");
        }else{
            for(std::map<std::string, Target>::iterator iter=device_map_ptr->begin(); iter != device_map_ptr->end(); iter++){
                printf("%s %d\n", iter->first.c_str(), iter->second);
            }
        }
    }
    ARM_COMPUTE_LOG_GRAPH_INFO("Before depthwise graph nodes: " + std::to_string(graph.nodes().size()) + "\n");
    // Apply IR mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::IR);
    ARM_COMPUTE_LOG_GRAPH_INFO("Before depthwise graph nodes: " + std::to_string(graph.nodes().size()) + "\n");
    for(auto& node: graph.nodes()){
        if(node != nullptr){
            ARM_COMPUTE_LOG_GRAPH_INFO(node->name());
        }
    }
    // Force target to all graph construct
    // TODO (COMPMID-2014) : Support heterogeneous execution
    Target forced_target = target;
    if(!is_target_supported(target))
    {
        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
    if(execution_type == ExecutionType::EXECUTION_TYPE_DEFAULT){
        force_target_to_graph(graph, forced_target);
        // Setup backend context
        // TODO (COMPMID-2014) : Setup all backends needed by the graph
        // Done by Chunwei Xia
        setup_requested_backend_context(ctx, forced_target);
    }else{
        force_target_to_graph(graph, forced_target, device_map_ptr);
        setup_neon_and_cl_backend_context(ctx);
        auto &nodes = graph.nodes();
        for(auto& node: nodes) {
            if(node){
                
                if(node->type()!=NodeType::Const && node->type()!=NodeType::Input)
                ARM_COMPUTE_LOG_GRAPH_INFO(node->name() + " " + get_target_string(node->assigned_target()));
            }
            
        }
    }
    // Configure all tensors, after the pass manager
    detail::configure_all_tensors(graph);
    
    // Apply backend mutating passes
    pm.run_type(graph, IGraphMutator::MutationType::Backend);

    // Perform topological sort
    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    // Validate all nodes
    detail::validate_all_nodes(graph);

    // Configure all nodes
    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    // Allocate const tensors and call accessors
    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);

    // Prepare graph
    // detail::prepare_all_tasks(workload);

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }

    // Prepare graph
    detail::prepare_all_tasks(workload);

    // Finalize Graph context
    ctx.finalize();
    
    std::once_flag flag;
    std::call_once(flag, detail::dump_graph_info, workload);
    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
}

void GraphManager::execute_graph(Graph &graph)
{
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    
    while(true)
    {
        // Call input accessors
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }

        // Run graph
        auto execution_type = it->second.ctx->config().execution_type;
        if(execution_type == ExecutionType::EXECUTION_TYPE_DEFAULT
            || execution_type == ExecutionType::EXECUTION_TYPE_SERIAL_HYBRID){
            detail::call_all_tasks(it->second);
        }else if(execution_type == ExecutionType::EXECUTION_TYPE_PARALLEL
            || execution_type == ExecutionType::EXECUTION_TYPE_ULAYER){
            detail::call_all_tasks_parallel(it->second);
        }
        
        // Call output accessors
        if(!detail::call_all_output_node_accessors(it->second))
        {
            return;
        }
    }
}

void GraphManager::execute_graph(Graph &graph, int loop_count)
{
    // Check if graph is finalized
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    // std::once_flag flag;
    // std::call_once(flag, detail::dump_graph_info, it->second);
    if(loop_count <= 0){
        return;
    }
    while(true)
    {
        // Call input accessors
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
        auto execution_type = it->second.ctx->config().execution_type;
        printf("Start warm up\n");
        if(execution_type == ExecutionType::EXECUTION_TYPE_DEFAULT
            || execution_type == ExecutionType::EXECUTION_TYPE_SERIAL_HYBRID){
            detail::call_all_tasks(it->second);
        }else if(execution_type == ExecutionType::EXECUTION_TYPE_PARALLEL
            || execution_type == ExecutionType::EXECUTION_TYPE_ULAYER){
            detail::call_all_tasks_parallel(it->second);
        }
        printf("End warm up\n");
        // const std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        std::chrono::time_point<std::chrono::steady_clock> clock_array[loop_count+1];
        
        auto start = std::chrono::steady_clock::now();
        clock_array[0] = start;
        // Run graph loop_count times
        // Run graph
        if(execution_type == ExecutionType::EXECUTION_TYPE_DEFAULT
            || execution_type == ExecutionType::EXECUTION_TYPE_SERIAL_HYBRID){
            for(int i=0; i<loop_count; ++i){
                detail::call_all_tasks(it->second);
                clock_array[i+1] = std::chrono::steady_clock::now();
            }
        }else if(execution_type == ExecutionType::EXECUTION_TYPE_PARALLEL
           || execution_type == ExecutionType::EXECUTION_TYPE_ULAYER){
            for(int i=0; i<loop_count; ++i){
                detail::call_all_tasks_parallel(it->second);
                clock_array[i+1] = std::chrono::steady_clock::now();
            }
        }
        auto end = std::chrono::steady_clock::now();
        // const std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
        // auto total_latency = std::chrono::duration<double, std::micro>(end - start).count();
        double latency_array[loop_count];
        double max_latency = 0, min_latency = 1e9;
        for(int i=0; i<loop_count; ++i){
            latency_array[i] = std::chrono::duration<double, std::micro>(clock_array[i+1]-clock_array[i]).count();
            if(latency_array[i] > max_latency){
                max_latency = latency_array[i];
            }if(latency_array[i] < min_latency){
                min_latency = latency_array[i];
            }
        }
        auto total_latency = std::chrono::duration<double, std::micro>(end-start).count();
        printf("Arm Compute library run %s %d times avg %f us, min %f us, max %f us", 
            graph.name().c_str(), loop_count, total_latency / loop_count, min_latency, max_latency);
        // Write profile info 
        detail::dump_workload_profile(it->second);
        
        // Call output accessors
        if(!detail::call_all_output_node_accessors(it->second))
        {
            return;
        }
    }
}


void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute