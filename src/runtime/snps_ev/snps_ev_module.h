/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file snps_ev_module.h
 * \brief Execution handling of snps_ev kernels
 */
#ifndef TVM_RUNTIME_SNPS_EV_SNPS_EV_MODULE_H_
#define TVM_RUNTIME_SNPS_EV_SNPS_EV_MODULE_H_

#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "../meta_data.h"

namespace tvm {
namespace runtime {
/*!
 * \brief create a snps_ev module for GPU devices from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "clbin", "cl"
 * \param fmap The map function information map of each function.
 */
Module SNPS_EVModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_SNPS_EV_SNPS_EV_MODULE_H_
