#ifndef ISAAC_DRIVER_HANDLE_H
#define ISAAC_DRIVER_HANDLE_H

#include <memory>

#include "isaac/defines.h"
#include "isaac/driver/common.h"
#include <iostream>
namespace isaac
{

namespace driver
{

  struct cu_event_t{
      operator bool() const { return first && second; }
      CUevent first;
      CUevent second;
  };

#define HANDLE_TYPE(CLTYPE, CUTYPE) Handle<CLTYPE, CUTYPE>

template<class CLType, class CUType>
class ISAACAPI Handle
{
private:
  static void _delete(CUcontext x);
  static void _delete(CUdeviceptr x);
  static void _delete(CUstream x);
  static void _delete(CUdevice);
  static void _delete(CUevent x);
  static void _delete(CUfunction);
  static void _delete(CUmodule x);
  static void _delete(cu_event_t x);

  static void release(cl_context x);
  static void release(cl_mem x);
  static void release(cl_command_queue x);
  static void release(cl_device_id x);
  static void release(cl_event x);
  static void release(cl_kernel x);
  static void release(cl_program x);

public:
  Handle(backend_type backend, bool take_ownership = true);
  bool operator==(Handle const & other) const;
  bool operator<(Handle const & other) const;
  CLType & cl();
  CLType const & cl() const;
  CUType & cu();
  CUType const & cu() const;
  ~Handle();

private:
DISABLE_MSVC_WARNING_C4251
  std::shared_ptr<CLType> cl_;
  std::shared_ptr<CUType> cu_;
RESTORE_MSVC_WARNING_C4251
private:
  backend_type backend_;
  bool has_ownership_;
};

}
}

#endif