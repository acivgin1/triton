#include "isaac/driver/kernel.h"
#include "isaac/driver/buffer.h"
#include <iostream>
#include <cstring>

namespace isaac
{

namespace driver
{

Kernel::Kernel(Program const & program, const char * name) : backend_(program.backend_), address_bits_(program.context().device().address_bits()), h_(backend_, true)
{
  switch(backend_)
  {
    case CUDA:
      cu_params_store_.reserve(64);
      cu_params_.reserve(64);
      cuda::check(dispatch::cuModuleGetFunction(&h_.cu(), program.h_.cu(), name));\
      break;
    case OPENCL:
      cl_int err;
      h_.cl() = dispatch::clCreateKernel(program.h_.cl(), name, &err);
      ocl::check(err);
      break;
    default:
      throw;
  }
}

void Kernel::setArg(unsigned int index, std::size_t size, void* ptr)
{
  switch(backend_)
  {
    case CUDA:
      if(index + 1> cu_params_store_.size())
      {
        cu_params_store_.resize(index+1);
        cu_params_.resize(index+1);
      }
      cu_params_store_[index].reset(malloc(size), free);
      memcpy(cu_params_store_[index].get(), ptr, size);
      cu_params_[index] = cu_params_store_[index].get();
      break;
    case OPENCL:
      ocl::check(dispatch::clSetKernelArg(h_.cl(), index, size, ptr));
      break;
    default:
      throw;
  }
}

void Kernel::setArg(unsigned int index, Buffer const & data)
{
  switch(backend_)
  {
    case CUDA:
    {
      setArg(index, sizeof(CUdeviceptr), (void*)&data.h_.cu()); break;
    }
    case OPENCL:
      ocl::check(dispatch::clSetKernelArg(h_.cl(), index, sizeof(cl_mem), (void*)&data.h_.cl()));
      break;
    default: throw;
  }
}

void Kernel::setSizeArg(unsigned int index, size_t N)
{
  switch(backend_)
  {
    case CUDA:
    {
      int NN = static_cast<cl_int>(N);
      setArg(index, sizeof(int), &NN);
      break;
    }
    case OPENCL:
    {
      cl_int NN = static_cast<cl_int>(N);
      setArg(index, 4, &NN);
      break;
    }

    default: throw;
  }
}

}

}
