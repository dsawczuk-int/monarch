/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// XPU bridge: translates NCCL API calls to oneCCL C API via dynamic loading.
// oneCCL's C API (v2) closely mirrors NCCL, so the mapping is mostly direct.

#include "bridge_xpu.h"
#include <dlfcn.h>
#include <cstring>
#include <iostream>

// oneCCL types used for function pointer declarations.
// We redeclare these locally to avoid depending on oneCCL headers at build time.
// TODO IT IS BETTER TO US HEADER
namespace oneccl_types {

typedef struct onecclComm* onecclComm_t;

#define ONECCL_UNIQUE_ID_BYTES 4096
typedef struct {
  union {
    struct {
      char legacy[512];
      char nccl[512];
      char any[2048];
    };
    char data[ONECCL_UNIQUE_ID_BYTES];
  };
} onecclUniqueId;

typedef enum {
  onecclSuccess = 0,
  onecclError = 1,
  onecclSystemError = 2,
  onecclInternalError = 3,
  onecclInvalidArgument = 4,
  onecclInvalidUsage = 5,
  onecclInProgress = 6,
  onecclFailureGPU = 7,
  onecclFailureCPU = 8,
  onecclAllocFailureCPU = 9,
  onecclAllocFailureGPU = 10,
  onecclPluginException = 11,
  onecclNotImplemented = 12
} onecclResult_t;

typedef enum {
  onecclInt8 = 0,
  onecclUint8 = 1,
  onecclInt32 = 2,
  onecclUint32 = 3,
  onecclInt64 = 4,
  onecclUint64 = 5,
  onecclFloat16 = 6,
  onecclFloat32 = 7,
  onecclFloat64 = 8,
  onecclBfloat16 = 9
} onecclDataType_t;

typedef enum {
  onecclSum = 0,
  onecclProd = 1,
  onecclMax = 2,
  onecclMin = 3,
  onecclAvg = 4,
  onecclNumOps = 5,
  onecclMaxRedOp = 0x7fffffff
} onecclRedOp_t;

typedef enum {
  onecclScalarDevice = 0,
  onecclScalarHostImmediate = 1
} onecclScalarResidence_t;

typedef struct onecclConfig {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char* netName;
  int splitShare;
  int multiThreaded;
  int plugin;  // onecclPluginType_t
} onecclConfig_t;

} // namespace oneccl_types

namespace nccl_sys {

// Convert oneCCL result to NCCL result.
// The first few error codes (0-5) have matching semantics.
static ncclResult_t convert_result(oneccl_types::onecclResult_t r) {
  switch (r) {
    case oneccl_types::onecclSuccess:
      return ncclSuccess;
    case oneccl_types::onecclSystemError:
      return ncclSystemError;
    case oneccl_types::onecclInternalError:
      return ncclInternalError;
    case oneccl_types::onecclInvalidArgument:
      return ncclInvalidArgument;
    case oneccl_types::onecclInvalidUsage:
      return ncclInvalidUsage;
    case oneccl_types::onecclInProgress:
      return ncclInProgress;
    default:
      return ncclInternalError;
  }
}

// Copy between our ncclUniqueId (4096 bytes) and oneCCL's onecclUniqueId (4096
// bytes). Both are 4096 bytes so this is a direct memcpy.
static void nccl_to_oneccl_id(
    const ncclUniqueId& nccl_id,
    oneccl_types::onecclUniqueId& oneccl_id) {
  static_assert(
      sizeof(ncclUniqueId) == sizeof(oneccl_types::onecclUniqueId),
      "UniqueId size mismatch");
  memcpy(&oneccl_id, &nccl_id, sizeof(oneccl_types::onecclUniqueId));
}

static void oneccl_to_nccl_id(
    const oneccl_types::onecclUniqueId& oneccl_id,
    ncclUniqueId& nccl_id) {
  static_assert(
      sizeof(ncclUniqueId) == sizeof(oneccl_types::onecclUniqueId),
      "UniqueId size mismatch");
  memcpy(&nccl_id, &oneccl_id, sizeof(ncclUniqueId));
}

struct OneCCLAPI {
  // Version and error handling
  oneccl_types::onecclResult_t (*onecclGetVersion_)(int*);
  oneccl_types::onecclResult_t (*onecclGetUniqueId_)(
      oneccl_types::onecclUniqueId*);
  const char* (*onecclGetErrorString_)(oneccl_types::onecclResult_t);
  const char* (*onecclGetLastError_)(oneccl_types::onecclComm_t);

  // Communicator creation and management
  oneccl_types::onecclResult_t (*onecclCommInitRank_)(
      oneccl_types::onecclComm_t*,
      size_t,
      oneccl_types::onecclUniqueId,
      int);
  oneccl_types::onecclResult_t (*onecclCommInitRankConfig_)(
      oneccl_types::onecclComm_t*,
      size_t,
      oneccl_types::onecclUniqueId,
      int,
      const oneccl_types::onecclConfig_t*);
  oneccl_types::onecclResult_t (*onecclCommSplit_)(
      oneccl_types::onecclComm_t,
      int,
      int,
      oneccl_types::onecclComm_t*,
      oneccl_types::onecclConfig_t*);
  oneccl_types::onecclResult_t (*onecclCommDestroy_)(
      oneccl_types::onecclComm_t);
  oneccl_types::onecclResult_t (*onecclCommCount_)(
      const oneccl_types::onecclComm_t,
      int*);
  oneccl_types::onecclResult_t (*onecclCommDevice_)(
      const oneccl_types::onecclComm_t,
      int*);
  oneccl_types::onecclResult_t (*onecclCommUserRank_)(
      const oneccl_types::onecclComm_t,
      int*);
  oneccl_types::onecclResult_t (*onecclSetDevice_)(uint32_t);

  // Collective communication
  oneccl_types::onecclResult_t (*onecclAllReduce_)(
      void*,
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      oneccl_types::onecclRedOp_t,
      oneccl_types::onecclComm_t,
      void*);
  oneccl_types::onecclResult_t (*onecclBroadcast_)(
      const void*,
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      int,
      oneccl_types::onecclComm_t,
      void*);
  oneccl_types::onecclResult_t (*onecclReduce_)(
      const void*,
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      oneccl_types::onecclRedOp_t,
      int,
      oneccl_types::onecclComm_t,
      void*);
  oneccl_types::onecclResult_t (*onecclAllGather_)(
      const void*,
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      oneccl_types::onecclComm_t,
      void*);
  oneccl_types::onecclResult_t (*onecclReduceScatter_)(
      const void*,
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      oneccl_types::onecclRedOp_t,
      oneccl_types::onecclComm_t,
      void*);

  // Point to point communication
  oneccl_types::onecclResult_t (*onecclSend_)(
      const void*,
      size_t,
      oneccl_types::onecclDataType_t,
      int,
      oneccl_types::onecclComm_t,
      void*);
  oneccl_types::onecclResult_t (*onecclRecv_)(
      void*,
      size_t,
      oneccl_types::onecclDataType_t,
      int,
      oneccl_types::onecclComm_t,
      void*);

  // Group calls
  oneccl_types::onecclResult_t (*onecclGroupStart_)();
  oneccl_types::onecclResult_t (*onecclGroupEnd_)();

  // User-defined reduction operators
  oneccl_types::onecclResult_t (*onecclRedOpCreatePreMulSum_)(
      oneccl_types::onecclRedOp_t*,
      void*,
      oneccl_types::onecclDataType_t,
      oneccl_types::onecclScalarResidence_t,
      oneccl_types::onecclComm_t);
  oneccl_types::onecclResult_t (*onecclRedOpDestroy_)(
      oneccl_types::onecclRedOp_t,
      oneccl_types::onecclComm_t);

  // Indicates whether initialization succeeded
  ncclResult_t init_result_;

  static OneCCLAPI* get();
};

namespace {

OneCCLAPI create_oneccl_api() {
  OneCCLAPI r{};
  r.init_result_ = ncclSuccess;

  // Try to open libccl.so (oneCCL shared library)
  void* handle = dlopen("libccl.so", RTLD_LAZY | RTLD_NOLOAD);
  if (!handle) {
    handle = dlopen("libccl.so", RTLD_LAZY);
  }
  if (!handle) {
    handle = dlopen("libccl.so.2", RTLD_LAZY);
  }

  if (!handle) {
    std::cerr << "[NCCL-SYS] Warning: Can't open oneCCL library: " << dlerror()
              << std::endl;
    r.init_result_ = ncclSystemError;
    return r;
  }

#define LOOKUP_ONECCL_ENTRY(name)                                              \
  r.name##_ = reinterpret_cast<decltype(r.name##_)>(dlsym(handle, #name));     \
  if (!r.name##_) {                                                            \
    std::cerr << "[NCCL-SYS] Warning: Can't find " << #name << ": "           \
              << dlerror() << std::endl;                                       \
    r.init_result_ = ncclSystemError;                                          \
    return r;                                                                  \
  }

  LOOKUP_ONECCL_ENTRY(onecclGetVersion)
  LOOKUP_ONECCL_ENTRY(onecclGetUniqueId)
  LOOKUP_ONECCL_ENTRY(onecclGetErrorString)
  LOOKUP_ONECCL_ENTRY(onecclGetLastError)
  LOOKUP_ONECCL_ENTRY(onecclCommInitRank)
  LOOKUP_ONECCL_ENTRY(onecclCommInitRankConfig)
  LOOKUP_ONECCL_ENTRY(onecclCommSplit)
  LOOKUP_ONECCL_ENTRY(onecclCommDestroy)
  LOOKUP_ONECCL_ENTRY(onecclCommCount)
  LOOKUP_ONECCL_ENTRY(onecclCommDevice)
  LOOKUP_ONECCL_ENTRY(onecclCommUserRank)
  LOOKUP_ONECCL_ENTRY(onecclSetDevice)
  LOOKUP_ONECCL_ENTRY(onecclAllReduce)
  LOOKUP_ONECCL_ENTRY(onecclBroadcast)
  LOOKUP_ONECCL_ENTRY(onecclReduce)
  LOOKUP_ONECCL_ENTRY(onecclAllGather)
  LOOKUP_ONECCL_ENTRY(onecclReduceScatter)
  LOOKUP_ONECCL_ENTRY(onecclSend)
  LOOKUP_ONECCL_ENTRY(onecclRecv)
  LOOKUP_ONECCL_ENTRY(onecclGroupStart)
  LOOKUP_ONECCL_ENTRY(onecclGroupEnd)
  LOOKUP_ONECCL_ENTRY(onecclRedOpCreatePreMulSum)
  LOOKUP_ONECCL_ENTRY(onecclRedOpDestroy)
#undef LOOKUP_ONECCL_ENTRY

  return r;
}

} // namespace

OneCCLAPI* OneCCLAPI::get() {
  static OneCCLAPI singleton = create_oneccl_api();
  return &singleton;
}

} // namespace nccl_sys

// Macro to get the oneCCL API and return early if initialization failed
#define GET_ONECCL_API(api_ptr)                                \
  nccl_sys::OneCCLAPI* api_ptr = nccl_sys::OneCCLAPI::get();   \
  if (api_ptr->init_result_ != ncclSuccess) {                  \
    return api_ptr->init_result_;                              \
  }

// Macro for functions that return const char*
#define GET_ONECCL_API_STR(api_ptr)                            \
  nccl_sys::OneCCLAPI* api_ptr = nccl_sys::OneCCLAPI::get();   \
  if (api_ptr->init_result_ != ncclSuccess) {                  \
    return "[NCCL-SYS] oneCCL library not initialized";        \
  }

// C API wrapper implementations - NCCL-compatible names backed by oneCCL
extern "C" {

// Version and error handling
ncclResult_t ncclGetVersion(int* version) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclGetVersion_(version));
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  GET_ONECCL_API(api);
  oneccl_types::onecclUniqueId oneccl_id;
  auto r = api->onecclGetUniqueId_(&oneccl_id);
  if (r == oneccl_types::onecclSuccess) {
    nccl_sys::oneccl_to_nccl_id(oneccl_id, *uniqueId);
  }
  return nccl_sys::convert_result(r);
}

const char* ncclGetErrorString(ncclResult_t result) {
  GET_ONECCL_API_STR(api);
  // Map NCCL result codes to oneCCL result codes for the error string lookup
  oneccl_types::onecclResult_t oneccl_result;
  switch (result) {
    case ncclSuccess:
      oneccl_result = oneccl_types::onecclSuccess;
      break;
    case ncclSystemError:
      oneccl_result = oneccl_types::onecclSystemError;
      break;
    case ncclInternalError:
      oneccl_result = oneccl_types::onecclInternalError;
      break;
    case ncclInvalidArgument:
      oneccl_result = oneccl_types::onecclInvalidArgument;
      break;
    case ncclInvalidUsage:
      oneccl_result = oneccl_types::onecclInvalidUsage;
      break;
    case ncclInProgress:
      oneccl_result = oneccl_types::onecclInProgress;
      break;
    default:
      oneccl_result = oneccl_types::onecclInternalError;
      break;
  }
  return api->onecclGetErrorString_(oneccl_result);
}

const char* ncclGetLastError(ncclComm_t comm) {
  GET_ONECCL_API_STR(api);
  return api->onecclGetLastError_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm));
}

// Communicator creation and management

ncclResult_t
ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank) {
  GET_ONECCL_API(api);
  oneccl_types::onecclUniqueId oneccl_id;
  nccl_sys::nccl_to_oneccl_id(commId, oneccl_id);
  auto r = api->onecclCommInitRank_(
      reinterpret_cast<oneccl_types::onecclComm_t*>(comm),
      static_cast<size_t>(nranks),
      oneccl_id,
      rank);
  return nccl_sys::convert_result(r);
}

ncclResult_t ncclCommInitAll(ncclComm_t* /*comm*/, int /*ndev*/, const int* /*devlist*/) {
  // Not implemented in oneCCL
  return ncclInternalError;
}

ncclResult_t ncclCommInitRankConfig(
    ncclComm_t* comm,
    int nranks,
    ncclUniqueId commId,
    int rank,
    ncclConfig_t* /*config*/) {
  GET_ONECCL_API(api);
  oneccl_types::onecclUniqueId oneccl_id;
  nccl_sys::nccl_to_oneccl_id(commId, oneccl_id);
  // Pass NULL for config - oneCCL config has different fields
  auto r = api->onecclCommInitRankConfig_(
      reinterpret_cast<oneccl_types::onecclComm_t*>(comm),
      static_cast<size_t>(nranks),
      oneccl_id,
      rank,
      nullptr);
  return nccl_sys::convert_result(r);
}

ncclResult_t ncclCommInitRankScalable(
    ncclComm_t* /*newcomm*/,
    int /*nranks*/,
    int /*myrank*/,
    int /*nId*/,
    ncclUniqueId* /*commIds*/,
    ncclConfig_t* /*config*/) {
  // No equivalent in oneCCL
  return ncclInternalError;
}

ncclResult_t ncclCommSplit(
    ncclComm_t comm,
    int color,
    int key,
    ncclComm_t* newcomm,
    ncclConfig_t* /*config*/) {
  GET_ONECCL_API(api);
  // Pass NULL for config
  auto r = api->onecclCommSplit_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      color,
      key,
      reinterpret_cast<oneccl_types::onecclComm_t*>(newcomm),
      nullptr);
  return nccl_sys::convert_result(r);
}

ncclResult_t ncclCommFinalize(ncclComm_t /*comm*/) {
  // Not implemented in oneCCL
  return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclCommDestroy_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm)));
}

ncclResult_t ncclCommAbort(ncclComm_t /*comm*/) {
  // Not implemented in oneCCL
  return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t /*comm*/, ncclResult_t* asyncError) {
  // Not implemented in oneCCL - report no async errors
  if (asyncError) {
    *asyncError = ncclSuccess;
  }
  return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclCommCount_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm), count));
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device) {
  GET_ONECCL_API(api);
  // oneCCL uses onecclCommDevice instead of ncclCommCuDevice
  return nccl_sys::convert_result(api->onecclCommDevice_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm), device));
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclCommUserRank_(
      reinterpret_cast<oneccl_types::onecclComm_t>(comm), rank));
}

// Memory management - no oneCCL equivalents

ncclResult_t ncclCommRegister(
    const ncclComm_t /*comm*/,
    void* /*buff*/,
    size_t /*size*/,
    void** /*handle*/) {
  return ncclInternalError;
}

ncclResult_t ncclCommDeregister(const ncclComm_t /*comm*/, void* /*handle*/) {
  return ncclInternalError;
}

ncclResult_t ncclMemAlloc(void** /*ptr*/, size_t /*size*/) {
  return ncclInternalError;
}

ncclResult_t ncclMemFree(void* /*ptr*/) {
  return ncclInternalError;
}

// Collective communication

ncclResult_t ncclAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  // oneCCL's onecclAllReduce takes non-const sendbuff
  return nccl_sys::convert_result(api->onecclAllReduce_(
      const_cast<void*>(sendbuff),
      recvbuff,
      count,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      static_cast<oneccl_types::onecclRedOp_t>(op),
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

ncclResult_t ncclBroadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int root,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclBroadcast_(
      sendbuff,
      recvbuff,
      count,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      root,
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

ncclResult_t ncclReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    int root,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclReduce_(
      sendbuff,
      recvbuff,
      count,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      static_cast<oneccl_types::onecclRedOp_t>(op),
      root,
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

ncclResult_t ncclAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    ncclDataType_t datatype,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclAllGather_(
      sendbuff,
      recvbuff,
      sendcount,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

ncclResult_t ncclReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclReduceScatter_(
      sendbuff,
      recvbuff,
      recvcount,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      static_cast<oneccl_types::onecclRedOp_t>(op),
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

// Point to point communication

ncclResult_t ncclSend(
    const void* sendbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclSend_(
      sendbuff,
      count,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      peer,
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

ncclResult_t ncclRecv(
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    int peer,
    ncclComm_t comm,
    xpuStream_t stream) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclRecv_(
      recvbuff,
      count,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      peer,
      reinterpret_cast<oneccl_types::onecclComm_t>(comm),
      static_cast<void*>(stream)));
}

// Group calls

ncclResult_t ncclGroupStart() {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclGroupStart_());
}

ncclResult_t ncclGroupEnd() {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclGroupEnd_());
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t* /*simInfo*/) {
  // No equivalent in oneCCL
  return ncclInternalError;
}

// User-defined reduction operators

ncclResult_t ncclRedOpCreatePreMulSum(
    ncclRedOp_t* op,
    void* scalar,
    ncclDataType_t datatype,
    ncclScalarResidence_t residence,
    ncclComm_t comm) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclRedOpCreatePreMulSum_(
      reinterpret_cast<oneccl_types::onecclRedOp_t*>(op),
      scalar,
      static_cast<oneccl_types::onecclDataType_t>(datatype),
      static_cast<oneccl_types::onecclScalarResidence_t>(residence),
      reinterpret_cast<oneccl_types::onecclComm_t>(comm)));
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  GET_ONECCL_API(api);
  return nccl_sys::convert_result(api->onecclRedOpDestroy_(
      static_cast<oneccl_types::onecclRedOp_t>(op),
      reinterpret_cast<oneccl_types::onecclComm_t>(comm)));
}

// XPU runtime-like functions

xpuError_t xpuSetDevice(int device) {
  nccl_sys::OneCCLAPI* api = nccl_sys::OneCCLAPI::get();
  if (api->init_result_ != ncclSuccess) {
    return xpuErrorInvalidValue;
  }
  auto r = api->onecclSetDevice_(static_cast<uint32_t>(device));
  return (r == oneccl_types::onecclSuccess) ? xpuSuccess : xpuErrorInvalidValue;
}

xpuError_t xpuStreamSynchronize(xpuStream_t /*stream*/) {
  // Stream synchronization on XPU should be done via SYCL queue::wait().
  // This stub returns success; callers using XPU should synchronize
  // their SYCL queues directly.
  return xpuSuccess;
}

} // extern "C"
