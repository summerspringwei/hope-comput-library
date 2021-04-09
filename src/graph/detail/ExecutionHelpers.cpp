/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include <fstream>
#include <sys/stat.h>

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/TypesUtils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
namespace graph
{
namespace detail
{
void validate_all_nodes(Graph &g)
{
    auto &nodes = g.nodes();

    // Create tasks
    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target                    assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            Status                    status          = backend.validate_node(*node);
            ARM_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
        }
    }
}

void configure_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && tensor->handle() == nullptr)
        {
            Target                         target  = tensor->desc().target;
            backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
            std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }
}

void allocate_all_input_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        Tensor *tensor = node.input(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_all_output_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *tensor = node.output(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_const_tensors(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
            }
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr && tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
        {
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order)
{
    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    // Reserve memory for tasks
    workload.tasks.reserve(node_order.size());

    // Create tasks
    for(auto &node_id : node_order)
    {
        auto node = g.node(node_id);
        if(node != nullptr)
        {
            Target                     assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            std::unique_ptr<IFunction> func            = backend.configure_node(*node, ctx);
            if(func != nullptr || is_utility_node(node))
            {
                workload.tasks.emplace_back(ExecutionTask(std::move(func), node));
            }
        }
    }

    // Add inputs and outputs
    for(auto &node : g.nodes())
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            continue;
        }
    }

    return workload;
}

void release_unused_tensors(Graph &g)
{
    for(auto &tensor : g.tensors())
    {
        if(tensor != nullptr && tensor->handle() != nullptr)
        {
            tensor->handle()->release_if_unused();
        }
    }
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const && node->num_outputs())
        {
            if(!node->output(0)->bound_edges().empty())
            {
                call_tensor_accessor(node->output(0));
            }
        }
    }
}

bool call_all_input_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.inputs), std::end(workload.inputs), [&](Tensor * input_tensor)
    {
        bool valid_input = (input_tensor != nullptr) && input_tensor->call_accessor();
        is_valid         = is_valid && valid_input;
    });
    return is_valid;
}

void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    for(auto &task : workload.tasks)
    {
        task.prepare();
        release_unused_tensors(*workload.graph);
    }
}
/**Map all the input and output tensors
 * @param[in] node: The node to map tensors
*/
void map_node(INode *node){
    for(auto edge_idx: node->input_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge->tensor()->desc().target != node->assigned_target()){
            edge->tensor()->handle()->map(true);
            printf("map in %p\n", (void*)(edge->tensor()->handle()->tensor().buffer()));
        }
    }
    for(auto edge_idx: node->output_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge->tensor()->desc().target != node->assigned_target()){
            edge->tensor()->handle()->map(true);
            edge->tensor()->handle()->tensor().buffer();
            printf("map out %p\n", (void*)(edge->tensor()->handle()->tensor().buffer()));
        }
    }
}

void unmap_node(INode *node){
    for(auto edge_idx: node->input_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge->tensor()->desc().target != node->assigned_target()){
            edge->tensor()->handle()->unmap();
        }
    }
    for(auto edge_idx: node->output_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge->tensor()->desc().target != node->assigned_target()){
            edge->tensor()->handle()->unmap();
        }
    }
}


void call_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);

    // Acquire memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }
    std::vector<CallStat> run_profile;
    // Execute tasks
    const std::chrono::time_point<std::chrono::high_resolution_clock> workload_start = std::chrono::high_resolution_clock::now();
    for(auto &task : workload.tasks)
    {
        CallStat stat;
        stat.name = task.node->name();
        stat.node_type_str = get_node_type_string(task.node->type());
        stat.target_str = get_target_string(task.node->assigned_target());
        const std::chrono::time_point<std::chrono::high_resolution_clock> task_start = std::chrono::high_resolution_clock::now();
        stat.start_micros = std::chrono::duration<double, std::micro>(task_start - workload_start).count();
        if(stat.name == "pool5"){
            printf("pool5\n");
        }
        // Blocking map input and output tensors if needed
        if(task.node->assigned_target() == Target::NEON){
            map_node(task.node);
        }
        printf("Start %s\n", stat.name.c_str());
        task();
        printf("End %s\n", stat.name.c_str());
        if(task.node->assigned_target()==Target::CL){
            // arm_compute::CLScheduler::get().sync();
            arm_compute::CLScheduler::get().wait();
        }
        // TODO(Chunwei Xia) Does not unmap the edge if not needed
        if(task.node->assigned_target() == Target::NEON){
            unmap_node(task.node);
        }
        
        const std::chrono::time_point<std::chrono::high_resolution_clock> task_end = std::chrono::high_resolution_clock::now();
        stat.end_micros = std::chrono::duration<double, std::micro>(task_end - workload_start).count();
        run_profile.push_back(stat);
    }
    workload.profiles.push_back(run_profile);
    // Release memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

bool call_all_output_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.outputs), std::end(workload.outputs), [&](Tensor * output_tensor)
    {
        bool valid_output = (output_tensor != nullptr) && output_tensor->call_accessor();
        is_valid          = is_valid && valid_output;
    });

    return is_valid;
}

bool fileExists(const char* filename)
{
    struct stat buf;
    if (stat(filename, &buf) != -1)
    {
        return true;
    }
    return false;
}

std::shared_ptr<std::map<std::string, Target>> read_device_map(const char * file_path){
    if(file_path == nullptr){
        return nullptr;
    }
    if (!fileExists(file_path)) {
        return nullptr;
    }
    std::ifstream infile(file_path, std::ifstream::in);
    std::string node;
    int64_t device;
    std::shared_ptr<std::map<std::string, Target>> device_map_ptr(new std::map<std::string, Target>());
    while (infile >> node >> device) {
        Target target;
        if(device==0){
            target = Target::NEON;
        }else if(device ==3){
            target = Target::CL;
        }else{
            target = Target::NEON;
        }
        device_map_ptr->insert(std::pair<std::string, Target>(node, target));
        printf("%s %ld\n", node.c_str(), device);
    }
    return device_map_ptr;
}



bool dump_workload_profile(ExecutionWorkload &workload){
    const std::string path = "/data/local/tmp/profile.txt";
    std::ofstream file_profile(path, std::ios::out);
    for(auto run_profile: workload.profiles){
        int count = 0;
        for(auto stat: run_profile){
            char buf[1024];
            sprintf(buf, "Iter %d: %s\t%s\t%s\t%lu\t%lu\t%lu\n", count,
                stat.name.c_str(), stat.node_type_str.c_str(), stat.target_str.c_str(), 
                stat.start_micros, stat.end_micros, (stat.end_micros - stat.start_micros));
            file_profile.write(buf, std::strlen(buf));
            count++;
        }
    }
    file_profile.flush();
    file_profile.close();
    return true;
}

} // namespace detail
} // namespace graph
} // namespace arm_compute
