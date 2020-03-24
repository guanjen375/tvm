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
 * \file snps_ev_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "snps_ev_common.h"
#include "snps_ev_module.h"

namespace tvm {
namespace runtime {

class SNPS_EVWrappedFunc {
 public:
  // initialize the snps_ev function.
  void Init(SNPS_EVModuleNode* m,
            ObjectPtr<Object> sptr,
            SNPS_EVModuleNode::KTRefEntry entry,
            std::string func_name,
            std::vector<size_t> arg_size,
            const std::vector<std::string>& thread_axis_tags)  {
    w_ = m->GetGlobalWorkspace().get();
    m_ = m;
    sptr_ = sptr;
    entry_ = entry;
    func_name_ = func_name;
    arg_size_ = arg_size;
    thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    CHECK(w_->context != nullptr) << "No snps_ev device";
    snps_ev::SNPS_EVThreadEntry* t = w_->GetThreadEntry();
    // get the kernel from thread local kernel table.
    if (entry_.kernel_id >= t->kernel_table.size()) {
      t->kernel_table.resize(entry_.kernel_id + 1);
    }
    const auto& e = t->kernel_table[entry_.kernel_id];
    cl_kernel kernel = e.kernel;
    if (kernel == nullptr || e.version != entry_.version) {
      kernel = m_->InstallKernel(w_, t, func_name_, entry_);
    }
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      SNPS_EV_CALL(clSetKernelArg(kernel, i, arg_size_[i], void_args[i]));
    }
    cl_command_queue queue = w_->GetQueue(t->context);
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(thread_axis_cfg_.work_dim());
    for (cl_uint i = 0; i < work_dim; ++i) {
      wl.work_size[i] *= wl.work_size[i + 3];
    }
    // launch kernel
    SNPS_EV_CALL(clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr,
        wl.work_size,
        wl.work_size + 3,
        0, nullptr, nullptr));
  }

 private:
  // global workspace.
  snps_ev::SNPS_EVWorkspace* w_;
  // The module
  SNPS_EVModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // global kernel id in the kernel table.
  SNPS_EVModuleNode::KTRefEntry entry_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // thread axis config
  ThreadAxisConfig thread_axis_cfg_;
};

SNPS_EVModuleNode::~SNPS_EVModuleNode() {
  {
    // free the kernel ids in global table.
    std::lock_guard<std::mutex> lock(workspace_->mu);
    for (auto& kv : kid_map_) {
      workspace_->free_kernel_ids.push_back(kv.second.kernel_id);
    }
  }
  // free the kernels
  for (cl_kernel k : kernels_) {
    SNPS_EV_CALL(clReleaseKernel(k));
  }
  if (program_) {
    SNPS_EV_CALL(clReleaseProgram(program_));
  }
}

const std::shared_ptr<snps_ev::SNPS_EVWorkspace>& SNPS_EVModuleNode::GetGlobalWorkspace() {
  return snps_ev::SNPS_EVWorkspace::Global();
}

PackedFunc SNPS_EVModuleNode::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  SNPS_EVWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    if (t.code == kHandle) {
      // specially store pointer type size in snps_ev driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      CHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  // initialize the wrapped func.
  f.Init(this, sptr_to_self, kid_map_.at(name),
         name, arg_size, info.thread_axis_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

void SNPS_EVModuleNode::SaveToFile(const std::string& file_name,
                                  const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  CHECK_EQ(fmt, fmt_)
      << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, data_);
}

void SNPS_EVModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
}

std::string SNPS_EVModuleNode::GetSource(const std::string& format) {
  if (format == fmt_) return data_;
  if (fmt_ == "cl") {
    return data_;
  } else {
    return source_;
  }
}

void SNPS_EVModuleNode::Init() {
  workspace_ = GetGlobalWorkspace();
  workspace_->Init();
  device_built_flag_.resize(workspace_->devices.size(), false);
  // initialize the kernel id, need to lock global table.
  std::lock_guard<std::mutex> lock(workspace_->mu);
  for (const auto& kv : fmap_) {
    const std::string& key = kv.first;
    KTRefEntry e;
    if (workspace_->free_kernel_ids.size() != 0) {
      e.kernel_id = workspace_->free_kernel_ids.back();
      workspace_->free_kernel_ids.pop_back();
    } else {
      e.kernel_id = workspace_->num_registered_kernels++;
    }
    e.version = workspace_->timestamp++;
    kid_map_[key] = e;
  }
}

cl_kernel SNPS_EVModuleNode::InstallKernel(snps_ev::SNPS_EVWorkspace* w,
                                          snps_ev::SNPS_EVThreadEntry* t,
                                          const std::string& func_name,
                                          const KTRefEntry& e) {
  std::lock_guard<std::mutex> lock(build_lock_);
  int device_id = t->context.device_id;
  if (!device_built_flag_[device_id]) {
    // create program
    if (fmt_ == "cl") {
      if (program_ == nullptr) {
        const char* s = data_.c_str();
        size_t len = data_.length();
        cl_int err;
        program_ = clCreateProgramWithSource(w->context, 1, &s, &len, &err);
        SNPS_EV_CHECK_ERROR(err);
      }
    } else if (fmt_ == "xclbin" || fmt_ == "awsxclbin" || fmt_ == "aocx") {
      const unsigned char* s = (const unsigned char *)data_.c_str();
      size_t len = data_.length();
      cl_int err;
      cl_device_id dev = w->devices[device_id];
      program_ = clCreateProgramWithBinary(w->context, 1, &dev, &len, &s, NULL, &err);
      SNPS_EV_CHECK_ERROR(err);
    } else {
      LOG(FATAL) << "Unknown snps_ev format " << fmt_;
    }
    // build program
    cl_int err;
    cl_device_id dev = w->devices[device_id];
    err = clBuildProgram(program_, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t len;
      std::string log;
      clGetProgramBuildInfo(
          program_, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
      log.resize(len);
      clGetProgramBuildInfo(
          program_, dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
      LOG(FATAL) << "snps_ev build error for device=" << dev << log;
    }
    device_built_flag_[device_id] = true;
  }
  // build kernel
  cl_int err;
  cl_kernel kernel = clCreateKernel(program_, func_name.c_str(), &err);
  SNPS_EV_CHECK_ERROR(err);
  t->kernel_table[e.kernel_id].kernel = kernel;
  t->kernel_table[e.kernel_id].version = e.version;
  kernels_.push_back(kernel);
  return kernel;
}

Module SNPS_EVModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  auto n = make_object<SNPS_EVModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

// Load module from module.
Module SNPS_EVModuleLoadFile(const std::string& file_name,
                            const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SNPS_EVModuleCreate(data, fmt, fmap, std::string());
}

Module SNPS_EVModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SNPS_EVModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadfile_snps_ev")
.set_body_typed(SNPS_EVModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadfile_snps_evbin")
.set_body_typed(SNPS_EVModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadbinary_snps_ev")
.set_body_typed(SNPS_EVModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
