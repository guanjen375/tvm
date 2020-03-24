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
 * \file snps_ev_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include "snps_ev_common.h"

namespace tvm {
namespace runtime {
namespace snps_ev {

SNPS_EVThreadEntry* SNPS_EVWorkspace::GetThreadEntry() {
  return SNPS_EVThreadEntry::ThreadLocal();
}

const std::shared_ptr<SNPS_EVWorkspace>& SNPS_EVWorkspace::Global() {
  static std::shared_ptr<SNPS_EVWorkspace> inst = std::make_shared<SNPS_EVWorkspace>();
  return inst;
}

void SNPS_EVWorkspace::SetDevice(TVMContext ctx) {
  GetThreadEntry()->context.device_id = ctx.device_id;
}

void SNPS_EVWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index< devices.size());
    return;
  }
  CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
  switch (kind) {
    case kExist: break;
    case kMaxThreadsPerBlock: {
      size_t value;
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index],  CL_DEVICE_MAX_WORK_GROUP_SIZE,
          sizeof(size_t), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      *rv = 1;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value;
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_LOCAL_MEM_SIZE,
          sizeof(cl_ulong), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion: return;
    case kDeviceName: {
      char value[128] = {0};
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_NAME,
          sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
    case kMaxClockRate: {
      cl_uint value;
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY,
          sizeof(cl_uint), &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint), &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      SNPS_EV_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
      *rv = ss.str();
      break;
    }
    case kGcnArch: return;
  }
}

void* SNPS_EVWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment, TVMType type_hint) {
  this->Init();
  CHECK(context != nullptr) << "No snps_ev device";
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(
      this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  SNPS_EV_CHECK_ERROR(err_code);
  return mptr;
}

void SNPS_EVWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  SNPS_EV_CALL(clFinish(this->GetQueue(ctx)));

  cl_mem mptr = static_cast<cl_mem>(ptr);
  SNPS_EV_CALL(clReleaseMemObject(mptr));
}

void SNPS_EVWorkspace::CopyDataFromTo(const void* from,
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMType type_hint,
                                     TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  if (IsSNPS_EVDevice(ctx_from) && IsSNPS_EVDevice(ctx_to)) {
    SNPS_EV_CALL(clEnqueueCopyBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        from_offset, to_offset, size, 0, nullptr, nullptr));
  } else if (IsSNPS_EVDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
    SNPS_EV_CALL(clEnqueueReadBuffer(
        this->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, from_offset, size,
        static_cast<char*>(to) + to_offset,
        0, nullptr, nullptr));
    SNPS_EV_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kDLCPU && IsSNPS_EVDevice(ctx_to)) {
    SNPS_EV_CALL(clEnqueueWriteBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>(to),
        CL_FALSE, to_offset, size,
        static_cast<const char*>(from) + from_offset,
        0, nullptr, nullptr));
    SNPS_EV_CALL(clFinish(this->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to snps_ev or between snps_ev";
  }
}

void SNPS_EVWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  SNPS_EV_CALL(clFinish(this->GetQueue(ctx)));
}

void* SNPS_EVWorkspace::AllocWorkspace(TVMContext ctx,
                                      size_t size,
                                      TVMType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(ctx, size);
}

void SNPS_EVWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  GetThreadEntry()->pool.FreeWorkspace(ctx, data);
}

typedef dmlc::ThreadLocalStore<SNPS_EVThreadEntry> SNPS_EVThreadStore;

SNPS_EVThreadEntry* SNPS_EVThreadEntry::ThreadLocal() {
  return SNPS_EVThreadStore::Get();
}

std::string GetPlatformInfo(
    cl_platform_id pid, cl_platform_info param_name) {
  size_t ret_size;
  SNPS_EV_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  SNPS_EV_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::string GetDeviceInfo(
    cl_device_id pid, cl_device_info param_name) {
  size_t ret_size;
  SNPS_EV_CALL(clGetDeviceInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  SNPS_EV_CALL(clGetDeviceInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_platform_id> GetPlatformIDs() {
  cl_uint ret_size;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  SNPS_EV_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> GetDeviceIDs(
    cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size;
  cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
  std::vector<cl_device_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  SNPS_EV_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
  return ret;
}

bool MatchPlatformInfo(
    cl_platform_id pid,
    cl_platform_info param_name,
    std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = GetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

void SNPS_EVWorkspace::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  if (context != nullptr) return;
  this->type_key = type_key;
  // matched platforms
  std::vector<cl_platform_id> platform_ids = snps_ev::GetPlatformIDs();
  if (platform_ids.size() == 0) {
    LOG(WARNING) << "No snps_ev platform matched given existing options ...";
    return;
  }
  this->platform_id = nullptr;
  for (auto platform_id : platform_ids) {
    if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
      continue;
    }
    std::vector<cl_device_id> devices_matched = snps_ev::GetDeviceIDs(platform_id, device_type);
    if ((devices_matched.size() == 0) && (device_type == "gpu")) {
      LOG(WARNING) << "Using CPU snps_ev device";
      devices_matched = snps_ev::GetDeviceIDs(platform_id, "cpu");
    }
    if (devices_matched.size() > 0) {
      this->platform_id = platform_id;
      this->platform_name = snps_ev::GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
      this->device_type = device_type;
      this->devices = devices_matched;
      break;
    }
  }
  if (this->platform_id == nullptr) {
    LOG(WARNING) << "No snps_ev device";
    return;
  }
  cl_int err_code;
  this->context = clCreateContext(
      nullptr, this->devices.size(), &(this->devices[0]),
      nullptr, nullptr, &err_code);
  SNPS_EV_CHECK_ERROR(err_code);
  CHECK_EQ(this->queues.size(), 0U);
  for (size_t i = 0; i < this->devices.size(); ++i) {
    cl_device_id did = this->devices[i];
    this->queues.push_back(
        clCreateCommandQueue(this->context, did, 0, &err_code));
    SNPS_EV_CHECK_ERROR(err_code);
  }
  initialized_ = true;
}

TVM_REGISTER_GLOBAL("device_api.snps_ev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = SNPS_EVWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
