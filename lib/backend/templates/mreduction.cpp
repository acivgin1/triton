#include <iostream>
#include "atidlas/backend/stream.h"
#include "atidlas/backend/templates/mreduction.h"
#include "atidlas/tools/to_string.hpp"
#include "atidlas/tools/make_map.hpp"
#include "atidlas/tools/make_vector.hpp"

namespace atidlas
{

mreduction_parameters::mreduction_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy): base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch_policy(_fetch_policy) { }


int mreduction::check_invalid_impl(cl::Device const &, expressions_tuple const &) const
{
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int mreduction::lmem_usage() const
{
  return p_.local_size_0*(p_.local_size_1+1);
}

std::string mreduction::generate_impl(unsigned int label, expressions_tuple const & expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width, std::vector<mapped_mreduction*> const & exprs) const
{
  using tools::to_string;


  kernel_generation_stream stream;

  char kprefix[10];
  fill_kernel_name(kprefix, label, "d");

  std::string arguments = "unsigned int M, unsigned int N, " ;
  for (const auto & e : exprs)
  {
    std::string numeric_type = numeric_type_to_string(lhs_most(e->array_expression().tree(), e->array_expression().root()).lhs.dtype);
    if (e->is_index_reduction())
    {
      arguments += e->process("__global unsigned int* #name_temp, ");
      arguments += e->process("__global " + to_string(numeric_type) + "* #name_temp_value,");
    }
    else
      arguments += e->process("__global " + to_string(numeric_type) + "* #name_temp, ");
  }

  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
  stream << "__kernel void " << kprefix << "0(" << arguments << generate_arguments("#scalartype", mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array0", "#scalartype #namereg = #pointer[#start];"},
                         {"array1", "#pointer += #start;"},
                         {"array2", "#pointer += #start1 + #start2*#ld; "
                                    "#ld *= #nldstride; "}}, expressions, mappings);

  unsigned int local_size_0_ld = p_.local_size_0+1;
  std::string local_size_0_ld_str = to_string(local_size_0_ld);

  for (const auto & e : exprs)
    stream << e->process("__local #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
  stream << "unsigned int lid1 = get_local_id(1);" << std::endl;
  stream << "unsigned int upper_bound_1 = ( M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << ";" << std::endl;
  stream << "for(unsigned int r = get_global_id(1); r < upper_bound_1; r += get_global_size(1)){" << std::endl;
  stream.inc_tab();

  for (const auto & e : exprs)
    stream << e->process("#scalartype #name_acc = " + neutral_element((e)->root_op()) + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  element_wise_loop_1D(stream, p_.fetch_policy, simd_width, "c", "N", "get_global_id(0)", "get_global_size(0)", [&](unsigned int simd_width)
  {
    std::string data_type = append_width("#scalartype",simd_width);

    for (const auto & e : exprs)
    {
      std::map<std::string, std::string> accessors;
      if(reduction_type_==REDUCE_COLUMNS)
      {
        accessors["array2"] = data_type + " #namereg = " + vload(simd_width, "c*#stride1", "#pointer + r*#ld")+";";
        accessors["repeat"] = data_type + " #namereg = " + vload(simd_width, "(c%#tuplearg0)*#stride", "#pointer + (r%#tuplearg1)*#stride ")+";";
      }
      else
      {
        accessors["array2"] = "#scalartype #namereg = #pointer[r*#stride1 + c*#ld];";
        accessors["repeat"] = "#scalartype #namereg = $VALUE{(r%#tuplearg0)*#stride, (c%#tuplearg1)*#stride};";
      }
      e->process_recursive(stream, PARENT_NODE_TYPE, accessors);
    }


    //Update accumulators
    std::vector<std::string> str(simd_width);
    if (simd_width==1)
      str[0] = "#namereg";
    else
      for (unsigned int a = 0; a < simd_width; ++a)
        str[a] = append_simd_suffix("#namereg.s",a);


    for (auto & elem : exprs)
      for (unsigned int a = 0; a < simd_width; ++a)
      {
        std::string value = elem->evaluate_recursive(LHS_NODE_TYPE, {{"array2", str[a]}, {"repeat", str[a]}, {"array0", "#namereg"}});
        if (elem->is_index_reduction())
          compute_index_reduction(stream, elem->process("#name_acc"), "c*"+to_string(simd_width) + to_string(a), elem->process("#name_acc_value"), value, elem->root_op());
        else
          compute_reduction(stream, elem->process("#name_acc"), value,elem->root_op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : exprs)
    stream << expr->process("#name_buf[lid1*" + local_size_0_ld_str + "+ lid0] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream <<  "if (lid0 < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : exprs)
    if (e->is_index_reduction())
      compute_index_reduction(stream, e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->root_op());
    else
      compute_reduction(stream,e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lid0 == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();
  if(p_.num_groups_0==1)
  {
    std::map<std::string, std::string> accessors;
    accessors["mreduction"] = "#name_buf[lid1*" + local_size_0_ld_str + "]";
    accessors["array1"] = "#pointer[r*#stride]";
    evaluate(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);
  }
  else
  {
    for (mapped_reduction const * e : exprs)
    {
      if (e->is_index_reduction())
        stream << e->process("#name_temp_value[r + M*get_group_id(0)] = #name_buf_value[lid1*" + local_size_0_ld_str + "];") << std::endl;
      stream << e->process("#name_temp[r + M*get_group_id(0)] = #name_buf[lid1*" + local_size_0_ld_str + "];") << std::endl;
    }
  }
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  if(p_.num_groups_0>1)
  {
  /////////////////////////////////////////
  ////////////// Kernel 2
  ////////////////////////////////////////

  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
  stream << "__kernel void " << kprefix << "1(" << arguments << generate_arguments("#scalartype", mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array0", "#scalartype #namereg = #pointer[#start];"},
                         {"array1", "#pointer += #start;"},
                         {"array2", "#pointer += #start1 + #start2*#ld; "
                                    "#ld *= #nldstride; "}}, expressions, mappings);

  for (const auto & e : exprs)
    stream << e->process("__local #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
  stream << "unsigned int lid1 = get_local_id(1);" << std::endl;
  stream << "unsigned int upper_bound_1 = ( M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << ";" << std::endl;
  stream << "for(unsigned int r = get_global_id(1); r < upper_bound_1; r += get_global_size(1)){" << std::endl;
  stream.inc_tab();

  for (const auto & e : exprs)
    stream << e->process("#scalartype #name_acc = " + neutral_element((e)->root_op()) + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "for(unsigned int c = get_local_id(0); c < " << p_.num_groups_0 << "; c += get_local_size(0)){" << std::endl;
  stream.inc_tab();

  for (mapped_reduction* e: exprs)
    compute_reduction(stream, e->process("#name_acc"), e->process("#name_temp[r + M*c]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : exprs)
    stream << expr->process("#name_buf[lid1*" + local_size_0_ld_str + "+ lid0] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream <<  "if (lid0 < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : exprs)
    if (e->is_index_reduction())
      compute_index_reduction(stream, e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->root_op());
    else
      compute_reduction(stream,e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lid0 == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();

  std::map<std::string, std::string> accessors;
  accessors["mreduction"] = "#name_buf[lid1*" + local_size_0_ld_str + "]";
  accessors["array1"] = "#pointer[r*#stride]";
  evaluate(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;
  }

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

std::vector<std::string> mreduction::generate_impl(unsigned int label, expressions_tuple const & expressions, std::vector<mapping_type> const & mappings) const
{
  std::vector<mapped_mreduction*> reductions;
  expressions_tuple::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;
  for (mit = mappings.begin(), sit = expressions.data().begin(); mit != mappings.end(); ++mit, ++sit)
  {
    array_expression const & first_expression = *expressions.data().front();
    std::vector<size_t> idx = filter_nodes(&is_reduction, first_expression, false);
    for (auto & elem : idx)
      reductions.push_back((mapped_mreduction*)(mit->at(mapping_key(elem, PARENT_NODE_TYPE)).get()));
  }

  std::vector<std::string> res;
  if (reduction_type_ && p_.simd_width>1)
  {
    res.push_back(generate_impl(label, expressions, mappings, p_.simd_width, reductions));
    res.push_back(generate_impl(label, expressions, mappings, 1, reductions));
  }
  else
    res.push_back(generate_impl(label, expressions, mappings, 1, reductions));
  return res;
}

mreduction::mreduction(mreduction::parameters_type const & parameters,
                                         mreduction::reduction_type rtype,
                                         binding_policy_t binding_policy) :
  base_impl<mreduction, mreduction_parameters>(parameters, binding_policy),
  reduction_type_(rtype){ }

std::vector<int_t> mreduction::input_sizes(expressions_tuple const & expressions)
{
  array_expression const & first_expression = *expressions.data().front();
  std::vector<std::size_t> idx = filter_nodes(&is_reduction, first_expression, false);
  std::pair<int_t, int_t> MN = matrix_size(lhs_most(first_expression.tree(), idx[0]));
  if(reduction_type_==REDUCE_COLUMNS)
    std::swap(MN.first,MN.second);
  return tools::make_vector<int_t>() << MN.first << MN.second;
}

void mreduction::enqueue(cl::CommandQueue & queue, std::vector<cl_ext::lazy_compiler> & programs, unsigned int label,  controller<expressions_tuple> const & controller)
{
  expressions_tuple const & expressions = controller.x();
  cl::Context const & context = expressions.context();

  std::vector<int_t> MN = input_sizes(expressions);
  std::vector<array_expression::node const *> reductions;
  for (const auto & e : expressions.data())
  {
    std::vector<size_t> reductions_idx = filter_nodes(&is_reduction, *e, false);
    for (auto & r : reductions_idx)
      reductions.push_back(&(e)->tree()[r]);
  }


  //Kernel
  int idx = 0;
  if(reduction_type_==REDUCE_COLUMNS && p_.simd_width>1 && requires_fallback(expressions))
    idx = 1;
  cl::Program & program = programs[idx].program();

  std::vector< cl::Buffer > tmp;
  std::vector< cl::Buffer > tmpidx;
  unsigned int dtype_size = size_of(lhs_most(expressions.data().front()->tree(), expressions.data().front()->root()).lhs.dtype);

  char kname[2][10];
  fill_kernel_name(kname[0], label, "d0");
  fill_kernel_name(kname[1], label, "d1");

  unsigned int nk = (p_.num_groups_0==1)?1:2;

  std::vector<cl::Kernel> kernels;
  for(unsigned int k = 0 ; k < nk ; ++k)
    kernels.push_back(cl::Kernel(program, kname[k]));

  for(unsigned int k = 0 ; k < nk ; ++k)
  {
    cl::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setArg(n_arg++, cl_uint(M));
    kernel.setArg(n_arg++, cl_uint(N));

    //Temporary buffers
    unsigned int i = 0;
    unsigned int j = 0;
    for (auto const & r : reductions)
    {
      if (is_index_reduction(r->op))
      {
        if (tmpidx.size() <= j)
          tmpidx.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, p_.num_groups_0*M*4));
        kernel.setArg(n_arg++, tmpidx[j]);
        j++;
      }
      if (tmp.size() <= i)
        tmp.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, p_.num_groups_0*M*dtype_size));
      kernel.setArg(n_arg++, tmp[i]);
      i++;
    }
    set_arguments(expressions, kernel, n_arg);
  }

  //NDRange
  cl::NDRange global[2] = { cl::NDRange(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1), cl::NDRange(p_.local_size_0, p_.local_size_1*p_.num_groups_1) };
  cl::NDRange local[2] = { cl::NDRange(p_.local_size_0, p_.local_size_1), cl::NDRange(p_.local_size_0, p_.local_size_1) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    controller.execution_options().enqueue_cache(queue, kernels[i], cl::NullRange, global[i], local[i]);
}

mreduction_rows::mreduction_rows(mreduction_parameters  const & parameters,
                                           binding_policy_t binding_policy):
  mreduction(parameters, REDUCE_ROWS, binding_policy){}

mreduction_rows::mreduction_rows(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind):
  mreduction(mreduction_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_ROWS, bind)
{}


mreduction_cols::mreduction_cols(mreduction::parameters_type  const & parameters,
                                           binding_policy_t binding_policy):
  mreduction(parameters, REDUCE_COLUMNS, binding_policy){}

mreduction_cols::mreduction_cols(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind):
  mreduction(mreduction_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_COLUMNS, bind)
{}

template class base_impl<mreduction, mreduction_parameters>;


}
