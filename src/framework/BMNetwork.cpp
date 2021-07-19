#include <stdio.h>
#include "BMNetwork.h"
namespace bm {

BMNetwork::BMNetwork(void *bmrt, const std::string &name): m_bmrt(bmrt), bmodelPath(name) {
    m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
    if (!bmrt_load_bmodel(m_bmrt, bmodelPath.c_str())) {
        BMLOG(FATAL, "load bmodel(%s) failed!", bmodelPath.c_str());
    }
    const char **names;
    int num;
    num = bmrt_get_network_number(m_bmrt);
    bmrt_get_network_names(m_bmrt, &names);
    for(int i=0;i < num; ++i) {
        m_network_names.push_back(names[i]);
    }
    free(names);

    auto net_name = m_network_names[0];
    m_netinfo = bmrt_get_network_info(bmrt, net_name.c_str());
    assert(m_netinfo->stage_num == 1);
    batchSize = 1;
    if(m_netinfo->input_num>0){
        batchSize = m_netinfo->stages[0].input_shapes[0].dims[0];
    }
}

void BMNetwork::showInfo()
{

}

TensorVec BMNetwork::createOutputTensors(){
    TensorVec tensors;
    auto innerTensors = new bm_tensor_t[m_netinfo->output_num];
    for(int i = 0; i < m_netinfo->output_num; ++i) {
        innerTensors[i].dtype = m_netinfo->output_dtypes[i];
        innerTensors[i].shape = m_netinfo->stages[0].output_shapes[i];
        innerTensors[i].st_mode = BM_STORE_1N;
        innerTensors[i].device_mem = bm_mem_null();
        tensors.push_back(std::make_shared<BMTensor>(m_handle, m_netinfo->output_names[i],
                                                    m_netinfo->output_scales[i], &innerTensors[i], i==0));
    }
    return tensors;
}

TensorVec BMNetwork::createInputTensors(){
    TensorVec tensors;
    auto innerTensors = new bm_tensor_t[m_netinfo->input_num];
    for(int i = 0; i < m_netinfo->input_num; ++i) {
        innerTensors[i].dtype = m_netinfo->input_dtypes[i];
        innerTensors[i].shape = m_netinfo->stages[0].input_shapes[i];
        innerTensors[i].st_mode = BM_STORE_1N;
        innerTensors[i].device_mem = bm_mem_null();
        tensors.push_back(std::make_shared<BMTensor>(m_handle, m_netinfo->input_names[i],
                                                    m_netinfo->input_scales[i], &innerTensors[i], i==0));
    }
    return tensors;
}

int BMNetwork::forward(TensorVec inTensors, TensorVec outTensors) {

    BM_ASSERT_EQ(m_netinfo->input_num, inTensors.size());
    BM_ASSERT_EQ(m_netinfo->output_num, outTensors.size());

    auto net_out_tensors = outTensors[0]->raw_tensor();
    auto net_in_tensors = inTensors[0]->raw_tensor();

    bool user_mem = false; // if false, bmrt will alloc mem every time.
    if (net_out_tensors->device_mem.size != 0) {
        // if true, bmrt don't alloc mem again.
        user_mem = true;
    }

    bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, net_in_tensors, m_netinfo->input_num,
                                  net_out_tensors, m_netinfo->output_num, user_mem, false);

    BM_ASSERT(ok, "bmrt_launch_tensor_ex failed");

#if 0
    for(int i = 0;i < m_netinfo->output_num; ++i) {
        auto tensor = net_out_tensors[i];
        BMLOG(INFO, "output_tensor [%d] size=%d", i, bmrt_tensor_device_size(&tensor));
    }
#endif

    return 0;
}

BMTensor::~BMTensor() {
    if (need_release){
        delete [] m_tensor;
    }
    if (m_raw_data != NULL) {
        delete [] m_raw_data;
        m_raw_data = NULL;
    }
    if (m_float_data != NULL) {
        delete [] m_float_data;
        m_float_data = NULL;
    }
}

size_t BMTensor::get_elem_num() const {
	return bmrt_shape_count(&m_tensor->shape);
}

size_t BMTensor::get_mem_size() const
{
    size_t count = bmrt_tensor_bytesize(m_tensor);
    if(m_tensor->st_mode == BM_STORE_4N){
        size_t N = m_tensor->shape.num_dims==0?1: m_tensor->shape.dims[0];
        count = count / N;
        size_t align_N = (N+3)/4*4;
        count = count * align_N;
    } else if(m_tensor->st_mode == BM_STORE_2N){
        size_t N = m_tensor->shape.num_dims==0?1: m_tensor->shape.dims[0];
        count = count / N;
        size_t align_N = (N+1)/2*2;
        count = count * align_N;
    }
    return count;
}

unsigned char *BMTensor::get_raw_data() {
    auto bytes = get_mem_size();
    if (m_raw_data == NULL) {
        m_raw_data = new unsigned char[bytes];
    }
    bm_status_t ret = bm_memcpy_d2s_partial(m_handle, m_raw_data, m_tensor->device_mem, bytes);
    BM_ASSERT_EQ(ret, BM_SUCCESS);
    return m_raw_data;
}

float *BMTensor::get_float_data() {
        if (m_tensor->dtype == BM_FLOAT32) {
            return (float*)get_raw_data();
        }else if (BM_INT8 == m_tensor->dtype || BM_UINT8 == m_tensor->dtype) {
            size_t elem_num = get_elem_num();
            if (m_float_data == nullptr){
               m_float_data = new float[elem_num];
            }
            get_raw_data();
            if(BM_INT8 == m_tensor->dtype){
		    auto int8_data = (int8_t*)m_raw_data;
		    for(size_t i = 0;i < elem_num; ++ i) {
			m_float_data[i] = int8_data[i] * m_scale;
		    }
            } else {
		    auto uint8_data = (uint8_t*)m_raw_data;
		    for(size_t i = 0;i < elem_num; ++ i) {
			m_float_data[i] = uint8_data[i] * m_scale;
		    }
            }
        }else{
            BMLOG(FATAL, "NOT support dtype=%d", m_tensor->dtype);
        }
	return m_float_data;
}

void BMTensor::dumpData(const char* filename) {
	if (m_tensor->dtype == BM_FLOAT32) {
float* data = get_float_data();
size_t elem_num = get_elem_num();
auto fp = fopen(filename, "w");
for(size_t i=0; i<elem_num; i++){
fprintf(fp, "%f\n", data[i]);
}
fclose(fp);

}
}

}
