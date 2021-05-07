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
#include <vector>
#include <thread>
#include <utility>
#include <sstream>
#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <sched.h>

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/TypesUtils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Log.h"

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
            const char * msg = node->name().empty() ? "empty": node->name().c_str();
            printf("name: %s ", msg);
            printf(" input tensor id: ");
            for(auto edge_idx: node->input_edges()){
                auto edge = node->graph()->edge(edge_idx);
                if(edge!=nullptr)
                printf(" %d ", edge->tensor_id());
            }printf(" output tensor id: ");
            
            for(auto edge_idx: node->output_edges()){
                auto edge = node->graph()->edge(edge_idx);
                if(edge!=nullptr)
                printf(" %d ", edge->tensor_id());
            }printf("\n");
            
            Target                    assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            Status                    status          = backend.validate_node(*node);
            if(bool(status)){
                const char* msg = node->name().empty() ? "empty_name": node->name().c_str();
                ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Valide %s ok\n", msg);
            }else{
                const char* msg = node->name().empty() ? "empty_name": node->name().c_str();
                ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Valide %s error\n", msg);
                continue;
            }
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
            printf("ConfigureTensor:%d \n", tensor->id());
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
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Configure %s\n", node->name().c_str());
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


/**Map all the input and output tensors
 * @param[in] node: The node to map tensors
*/
int map_node(INode *node){
    int count = 0;
    for(auto edge_idx: node->input_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge == nullptr || edge->tensor()==nullptr){
            continue;
        }
        if(edge->tensor()->desc().target != node->assigned_target()){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("map input tensor id :%d buff: %p is subtensor: %d tensor target: %s node name %s target: %s\n", 
                edge->tensor()->id(),
                (void*)(edge->tensor()->handle()->tensor().buffer()),
                edge->tensor()->handle()->is_subtensor(),
                get_target_string(edge->tensor()->desc().target).c_str(), 
                node->name().c_str(),
                get_target_string(node->assigned_target()).c_str());
            edge->tensor()->handle()->map(true);
            count++;            
        }
    }
    for(auto edge_idx: node->output_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge == nullptr || edge->tensor()==nullptr){
            continue;
        }
        if(edge->tensor()->desc().target != node->assigned_target()){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("map output tensor id:%d buff: %p is subtensor: %d tensor target: %s node name %s target: %s\n", 
                edge->tensor()->id(),
                (void*)(edge->tensor()->handle()->tensor().buffer()),
                edge->tensor()->handle()->is_subtensor(),
                get_target_string(edge->tensor()->desc().target).c_str(), 
                node->name().c_str(),
                get_target_string(node->assigned_target()).c_str());
            edge->tensor()->handle()->map(true);
            count++;
            
        }
    }
    return count;
}

int unmap_node(INode *node){
    int count = 0;
    for(auto edge_idx: node->input_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge == nullptr || edge->tensor()==nullptr){
            continue;
        }
        if(edge->tensor()->desc().target != node->assigned_target()){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("unmap input tensor id :%d buff: %p is subtensor: %d tensor target: %s node name %s target: %s\n", 
                edge->tensor()->id(),
                (void*)(edge->tensor()->handle()->tensor().buffer()),
                edge->tensor()->handle()->is_subtensor(),
                get_target_string(edge->tensor()->desc().target).c_str(), 
                node->name().c_str(),
                get_target_string(node->assigned_target()).c_str());
            edge->tensor()->handle()->unmap();
            count++;
        }
    }
    for(auto edge_idx: node->output_edges()){
        auto edge = node->graph()->edge(edge_idx);
        if(edge == nullptr || edge->tensor()==nullptr){
            continue;
        }
        if(edge->tensor()->desc().target != node->assigned_target()){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("unmap output tensor id :%d buff: %p is subtensor: %d tensor target: %s node name %s target: %s\n", 
                edge->tensor()->id(),
                (void*)(edge->tensor()->handle()->tensor().buffer()),
                edge->tensor()->handle()->is_subtensor(),
                get_target_string(edge->tensor()->desc().target).c_str(), 
                node->name().c_str(),
                get_target_string(node->assigned_target()).c_str());
            edge->tensor()->handle()->unmap();
            count++;
        }
    }
    return count;
}


void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    for(auto &task : workload.tasks)
    {
        printf("prepare: %s %d\n", task.node->name().c_str(), task.node->type());
        // Here we only need to map the weight tensor
        if(task.node->type()==NodeType::Const || task.node->type()==NodeType::DepthwiseConvolutionLayer || task.node->type()==NodeType::ConvolutionLayer)
        map_node(task.node);
        task.prepare();
        if(task.node->type()==NodeType::Const || task.node->type()==NodeType::DepthwiseConvolutionLayer || task.node->type()==NodeType::ConvolutionLayer)
        unmap_node(task.node);
        release_unused_tensors(*workload.graph);
        printf("pfinish: %s\n", task.node->name().c_str());
    }
}


void call_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);
    auto execution_type = workload.ctx->config().execution_type;
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
        // Blocking map input and output tensors if needed
        const std::chrono::time_point<std::chrono::high_resolution_clock> task_start = std::chrono::high_resolution_clock::now();
        if((execution_type == ExecutionType::EXECUTION_TYPE_SERIAL_HYBRID)
            && task.node->assigned_target() == Target::NEON){
            map_node(task.node);
        }
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Start %s\n", task.node->name().c_str());
        task();
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("End %s\n", task.node->name().c_str());
        if(task.node->assigned_target()==Target::CL){
            // arm_compute::CLScheduler::get().sync();
            arm_compute::CLScheduler::get().wait();
        }
        // TODO(Chunwei Xia) Does not unmap the edge if not needed
        if((execution_type == ExecutionType::EXECUTION_TYPE_SERIAL_HYBRID)
            && task.node->assigned_target() == Target::NEON){
            unmap_node(task.node);
        }
    
        const std::chrono::time_point<std::chrono::high_resolution_clock> task_end = std::chrono::high_resolution_clock::now();

        CallStat stat{
            task.node->name(),
            get_node_type_string(task.node->type()),
            get_target_string(task.node->assigned_target()),
            {},
            static_cast<uint64_t>(std::chrono::duration<double, std::micro>(task_start - workload_start).count()),
            static_cast<uint64_t>(std::chrono::duration<double, std::micro>(task_end - workload_start).count())
        };
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

bool ready_to_execute(INode* node_ptr){
    if(node_ptr == nullptr){
        ARM_COMPUTE_ERROR_ON("Task's node ptr is null\n");
        return false;
    }
    
    for(auto edge_idx: node_ptr->input_edges()){
        auto edge = node_ptr->graph()->edge(edge_idx);
        if(edge == nullptr){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Node %s edge %d is null\n", node_ptr->name().c_str(), edge_idx);
            continue;
        }
        if(edge->producer() == nullptr){
            continue;
        }
        if(edge->producer()->type() == NodeType::Const 
            || edge->producer()->type() == NodeType::Input){
            continue;
        }if(edge->producer()->type() == NodeType::SplitLayer){
            // For parent being SplitLayer, we need to call recursively
            // to find whether all of it's precedence are executed
            if(!ready_to_execute(edge->producer())){
                return false;
            }
        }
        if(!(edge->producer()->executed())){
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s precedent %s is not ready\n", 
                node_ptr->name().c_str(), edge->producer()->name().c_str());
            return false;
        }
    }
    // All of it's precedence are executed
    if(node_ptr->type()==NodeType::SplitLayer){
        node_ptr->set_executed(true);
    }
    return true;
}

void run(std::vector<ExecutionTask*>& device_tasks, 
        std::vector<CallStat>& run_profile, 
        std::chrono::time_point<std::chrono::high_resolution_clock> workload_start,
        std::vector<int> cpu_core_list, 
        pthread_mutex_t* fast_mutex, pthread_cond_t* fast_cond){
    // Check parameters
    if(device_tasks.size() == 0){
        return;
    }
    
    const int num_of_tasks = device_tasks.size();
    auto target = device_tasks[0]->node->assigned_target();
    std::string forward_str;
    if(target == Target::NEON){
        forward_str = "CPU";
    }else{
        forward_str = "OpenCL";
    }
    // printf("Start execute on %s units size %d coreId %d\n",
    //         forward_str.c_str(), num_of_tasks, sched_getcpu());
    // printf("OPM_NUM_THREADS: %d\n", omp_get_num_threads());
    printf("Forward %s on cpu cores:", forward_str.c_str());
    for(auto core_id: cpu_core_list){
        printf(" %d", core_id);
    }printf("\n");
    int executed_task_count = num_of_tasks;
    double map_overhead = 0;
    int map_count = 0;
    while(executed_task_count--){
        ExecutionTask *current_task = nullptr;
        while(true){
            // Count for how many times try to executed a task that has not executed
            int count = 0;
            bool find_ready_task = false;
            for(auto task: device_tasks) {
                if(task->node->executed()){
                   continue;
                }
                count++;
                if(count > 10){
                    break;
                }
                if(ready_to_execute(task->node)){
                    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s in %s is ready\n", task->node->name().c_str(), forward_str.c_str());
                    find_ready_task = true;
                    current_task = task;
                    break;
                }else{
                    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s in %s is not ready\n", task->node->name().c_str(), forward_str.c_str());
                }
            }
            if(find_ready_task){
                break;
            }else{
                // First notify other thread
                pthread_cond_signal(fast_cond);
                // Second try get mutex
                while(pthread_mutex_trylock(fast_mutex)!=0){    
                }
                // If get fast_cond that release mutex
                pthread_cond_wait(fast_cond, fast_mutex);
                // Automatically get mutex after notified, therefore we need to unlock it
                pthread_mutex_unlock(fast_mutex);
            }
        }
        if(current_task == nullptr){
            continue;
        }
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Start %s on %s\n", current_task->node->name().c_str(), forward_str.c_str());
        
        // Blocking map input and output tensors if needed
        if(target == Target::NEON){
            auto map_start = std::chrono::high_resolution_clock::now();
            map_count += map_node(current_task->node);
            auto map_end = std::chrono::high_resolution_clock::now();
            map_overhead += std::chrono::duration<double, std::micro>(map_end - map_start).count();
        }
        auto task_start = std::chrono::high_resolution_clock::now();
        (*current_task)();
        if(target == Target::NEON){
            (*current_task).node->set_executed(true);
        }
        else{
            bool is_successor_on_cpu = false;
            // If its successor's precessor is on CPU, then it must start as soon as possible
            for(auto edge_idx: current_task->node->output_edges()){
                auto edge = current_task->node->graph()->edge(edge_idx);
                if(edge !=nullptr && edge->consumer()->assigned_target() == Target::NEON){
                    is_successor_on_cpu = true;
                    break;
                }else{
                    for(auto edge_in_idx: edge->consumer()->input_edges()){
                        auto edge_in = current_task->node->graph()->edge(edge_in_idx);
                        if(edge_in != nullptr &&
                            edge_in->producer() != nullptr &&
                            edge_in->producer()->assigned_target() == Target::NEON){
                            is_successor_on_cpu = true;
                            break;  
                        }
                    }
                    if(is_successor_on_cpu){
                        break;
                    }
                }
            }
            
            if(is_successor_on_cpu){
                arm_compute::CLScheduler::get().wait();
            }
            (*current_task).node->set_executed(true);
        }
        if(target == Target::NEON){
            auto map_start = std::chrono::high_resolution_clock::now();
            // map_count += unmap_node(current_task->node);
            auto map_end = std::chrono::high_resolution_clock::now();
            map_overhead += std::chrono::duration<double, std::micro>(map_end - map_start).count();
        }
        // Notify other thread
        pthread_cond_signal(fast_cond);
        auto task_end = std::chrono::high_resolution_clock::now();
        // node_attr = '\"{}: {}\" [shape=box,style=filled,color={}];\n'.format(index, short_op_name, "red")
        // if(forward_str=="CPU"){
        //     printf(" \"%s\" [shape=box,style=filled,color=red]", current_task->node->name().c_str());
        // }else{
        //    printf(" \"%s\" [shape=box,style=filled,color=green]", current_task->node->name().c_str());
        // }
        
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("End %s on %s\n", current_task->node->name().c_str(), forward_str.c_str());
        CallStat stat{
            current_task->node->name(),
            get_node_type_string(current_task->node->type()),
            get_target_string(current_task->node->assigned_target()),
            {},
            static_cast<uint64_t>(std::chrono::duration<double, std::micro>(task_start - workload_start).count()),
            static_cast<uint64_t>(std::chrono::duration<double, std::micro>(task_end - workload_start).count())
        };
        pthread_mutex_lock(fast_mutex);
        run_profile.push_back(stat);
        pthread_mutex_unlock(fast_mutex);
    }
    printf("Map cound %d overhead: %f\n", map_count, map_overhead);
}

void call_all_tasks_parallel(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);
    auto execution_type = workload.ctx->config().execution_type;
    if(execution_type != ExecutionType::EXECUTION_TYPE_PARALLEL && execution_type != ExecutionType::EXECUTION_TYPE_ULAYER){
        return ;
    }
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
    std::chrono::time_point<std::chrono::high_resolution_clock> workload_start = std::chrono::high_resolution_clock::now();
    
    std::vector<ExecutionTask*> cpu_queue, gpu_queue;
    for(auto &task : workload.tasks) {
        task.node->set_executed(false);
        if(task.node->assigned_target() == Target::NEON){
            cpu_queue.push_back(&task);
        }else{
            gpu_queue.push_back(&task);
        }
    }
    pthread_mutex_t fast_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fast_cond = PTHREAD_COND_INITIALIZER;
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s\n", "----- CPU tasks-----");
    for(auto* task: cpu_queue){
        ARM_COMPUTE_UNUSED(task);
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s\n", task->node->name().c_str());
    }
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s\n", "----- GPU tasks-----");
    for(auto* task: gpu_queue){
        ARM_COMPUTE_UNUSED(task);
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s\n", task->node->name().c_str());
    }
// void run(std::vector<ExecutionTask*>& device_tasks, 
//         std::vector<CallStat>& run_profile, 
//         std::chrono::time_point<std::chrono::high_resolution_clock> workload_start,
//         std::vector<int> cpu_core_list, 
//         pthread_mutex_t* fast_mutex, pthread_cond_t* fast_cond){
    std::vector<int> core_list = {0};
    std::thread cpu_thread(run, std::ref(cpu_queue), std::ref(run_profile), workload_start, core_list, &fast_mutex, &fast_cond);
    std::thread gpu_thread(run, std::ref(gpu_queue), std::ref(run_profile), workload_start, core_list, &fast_mutex, &fast_cond);

    cpu_thread.join();
    gpu_thread.join();
    
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
    ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Get operation device map:%d\n", 10);
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
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("%s %ld\n", node.c_str(), device);
    }
    return device_map_ptr;
}

bool dump_graph_info(ExecutionWorkload& workload){
    
    const std::string path = "/data/local/tmp/"+(workload.graph->name());
    std::ofstream file_info(path, std::ios::out);\
    
    std::stringstream ss;
    for(auto& task: workload.tasks){
        auto node_ptr= task.node;
        ss << node_ptr->name() << " ";
        for(auto edge_idx: node_ptr->input_edges()){
            auto edge = node_ptr->graph()->edge(edge_idx);
            if(edge == nullptr){
                continue;
            }
            if(edge->producer() == nullptr){
                continue;
            }if( edge->producer()->type()==NodeType::Const){
                continue;
            }
            auto shape = edge->tensor()->desc().shape;
            for(uint32_t i=0; i<shape.num_dimensions()-1; ++i){
                ss<<shape[i]<<",";
            }ss<<shape[shape.num_dimensions()-1]<<"@";
            ss<<edge->tensor()->id()<<";";
        }ss<<" ";
        for(auto edge_idx: node_ptr->output_edges()){
            auto edge = node_ptr->graph()->edge(edge_idx);
            auto shape = edge->tensor()->desc().shape;
            for(uint32_t i=0; i<shape.num_dimensions()-1; ++i){
                ss<<shape[i]<<",";
            }ss<<shape[shape.num_dimensions()-1]<<"@";
            ss<<edge->tensor()->id()<<";";
        }
        ss <<"\n";
    }
    file_info.write(ss.str().c_str(), ss.str().length());
    file_info.flush();
    file_info.close();
    return true;
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
                 (stat.end_micros - stat.start_micros), 
                 stat.start_micros, stat.end_micros);
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
