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
#ifndef ARM_COMPUTE_GRAPH_TYPE_LOADER_H
#define ARM_COMPUTE_GRAPH_TYPE_LOADER_H

#include "arm_compute/graph/Types.h"

#include <istream>

namespace arm_compute
{
/** Converts a string to a strong types enumeration @ref DataLayout
 *
 * @param[in] name String to convert
 *
 * @return Converted DataLayout enumeration
 */
arm_compute::DataLayout data_layout_from_name(const std::string &name);

/** Input Stream operator for @ref DataLayout
 *
 * @param[in]  stream      Stream to parse
 * @param[out] data_layout Output data layout
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, arm_compute::DataLayout &data_layout)
{
    std::string value;
    stream >> value;
    data_layout = data_layout_from_name(value);
    return stream;
}

namespace graph
{
/** Converts a string to a strong types enumeration @ref Target
 *
 * @param[in] name String to convert
 *
 * @return Converted Target enumeration
 */
Target target_from_name(const std::string &name);

/** Input Stream operator for @ref Target
 *
 * @param[in]  stream Stream to parse
 * @param[out] target Output target
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, Target &target)
{
    std::string value;
    stream >> value;
    target = target_from_name(value);
    return stream;
}

ExecutionType execution_type_from_name(std::string &name);

/** Input Stream operator for @ref ExecutionType
 *
 * @param[in]  stream Stream to parse
 * @param[out] execution_type Output execution_type
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, ExecutionType &execution_type)
{
    std::string value;
    stream >> value;
    execution_type = execution_type_from_name(value);
    return stream;
}

/** Converts a string to a strong types enumeration @ref ConvolutionMethod
 *
 * @param[in] name String to convert
 *
 * @return Converted Target enumeration
 */
ConvolutionMethod Convolution_method_from_name(const std::string &name);

/** Input Stream operator for @ref ConvolutionMethod
 *
 * @param[in]  stream Stream to parse
 * @param[out] target Output target
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, ConvolutionMethod &target)
{
    std::string value;
    stream >> value;
    target = Convolution_method_from_name(value);
    return stream;
}

/** Converts a string to a strong types enumeration @ref DepthwiseConvolutionMethod
 *
 * @param[in] name String to convert
 *
 * @return Converted Target enumeration
 */
DepthwiseConvolutionMethod depthwise_convolution_method_from_name(const std::string &name);

/** Input Stream operator for @ref DepthwiseConvolutionMethod
 *
 * @param[in]  stream Stream to parse
 * @param[out] target Output target
 *
 * @return Updated stream
 */
inline ::std::istream &operator>>(::std::istream &stream, DepthwiseConvolutionMethod &target)
{
    std::string value;
    stream >> value;
    target = depthwise_convolution_method_from_name(value);
    return stream;
}



} // namespace graph
} // namespace arm_compute
#endif /* ARM_COMPUTE_GRAPH_TYPE_LOADER_H */
