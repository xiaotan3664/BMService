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

static std::string shape_to_str(const bm_shape_t& shape) {
    std::string str ="[ ";
    for(int i=0; i<shape.num_dims; i++){
        str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
}

void BMNetwork::showInfo()
{
    const char* dtypeMap[] = {
        "FLOAT32",
        "FLOAT16",
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
    };
    BMLOG(INFO, "NetName: %s", m_netinfo->name);
    for(int i=0; i<m_netinfo->input_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[0].input_shapes[i]);
        BMLOG(INFO, "  Input %d) '%s' shape=%s dtype=%s scale=%g",
              i,
              m_netinfo->input_names[i],
              shapeStr.c_str(),
              dtypeMap[m_netinfo->input_dtypes[i]],
              m_netinfo->input_scales[i]);
    }
    for(int i=0; i<m_netinfo->output_num; i++){
        auto shapeStr = shape_to_str(m_netinfo->stages[0].output_shapes[i]);
        BMLOG(INFO, "  Output %d) '%s' shape=%s dtype=%s scale=%g",
              i,
              m_netinfo->output_names[i],
              shapeStr.c_str(),
              dtypeMap[m_netinfo->output_dtypes[i]],
              m_netinfo->output_scales[i]);
    }

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

    int static_batch_size = m_netinfo->stages[0].input_shapes[0].dims[0];
    int runtime_batch_size = net_in_tensors[0].shape.dims[0];
    if (!m_netinfo->is_dynamic && runtime_batch_size < static_batch_size)
    {
        // Static model with input batch size smaller than model n size
        // Use model n to do forwarding
        BMLOG(DEBUG, "override batch size from %d to %d", runtime_batch_size, static_batch_size);
        for (int i = 0; i < m_netinfo->input_num; ++i)
            net_in_tensors[i].shape.dims[0] = static_batch_size;
        for (int i = 0; i < m_netinfo->output_num; ++i)
            net_out_tensors[i].shape.dims[0] = static_batch_size;
    }

    bool user_mem = false; // if false, bmrt will alloc mem every time.
    if (net_out_tensors->device_mem.size != 0) {
        // if true, bmrt don't alloc mem again.
        user_mem = true;
    }

    bool ok=bmrt_launch_tensor_ex(m_bmrt, m_netinfo->name, net_in_tensors, m_netinfo->input_num,
                                  net_out_tensors, m_netinfo->output_num, user_mem, false);

    if (!m_netinfo->is_dynamic && runtime_batch_size < static_batch_size)
    {
        // Set runtime batch size for static model
        for (int i = 0; i < m_netinfo->output_num; ++i)
            net_out_tensors[i].shape.dims[0] = runtime_batch_size;
    }

    BM_ASSERT(ok, "bmrt_launch_tensor_ex failed");
    bm_thread_sync(m_handle);

#if 0
    for(int i = 0;i < m_netinfo->output_num; ++i) {
        auto tensor = net_out_tensors[i];
        BMLOG(INFO, "output_tensor [%d] size=%d", i, bmrt_tensor_device_size(&tensor));
    }
#endif

    return ok;
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

bool BMTensor::fill_device_mem(const void *data, size_t len, size_t mem_offset)
{
    auto mem = get_device_mem();
    return bm_memcpy_s2d_partial_offset(m_handle, *mem, (void*)data, len, mem_offset) == BM_SUCCESS;
}

size_t BMTensor::get_dtype_len() const {
    auto t = m_tensor->dtype;
    if(t == BM_FLOAT32 || t == BM_INT32 || t== BM_UINT32){
        return 4;
    } else if(t == BM_UINT16 || t==BM_INT16 || t==BM_FLOAT16){
        return 2;
    } else if(t == BM_UINT8 || t == BM_UINT8){
        return 1;
    } else {
        BMLOG(FATAL, "Not support dtype=%d", t);
    }
}

ssize_t BMTensor::partial_shape_count(size_t begin, size_t end)
{
    size_t count = 1;
    for(size_t i=begin; i<=end; i++){
        count *= shape(i);
    }
    return count;
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
    if(m_raw_size < bytes) {
        delete [] m_raw_data;
        m_raw_data = nullptr;
        m_raw_size = 0;
    }
    if (m_raw_data == nullptr) {
        m_raw_data = new unsigned char[bytes];
        m_raw_size = bytes;
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

size_t BMTensor::fill_host_mem(void *ptr, size_t len)
{
    auto bytes = get_mem_size();
    size_t real_len = bytes>len? len: bytes;
    bm_status_t ret = bm_memcpy_d2s_partial(m_handle, ptr, m_tensor->device_mem, real_len);
    BM_ASSERT_EQ(ret, BM_SUCCESS);
    return real_len;
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
