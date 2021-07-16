#include "BMNetwork.h"
namespace bm {

BMNetwork::BMNetwork(void *bmrt, const std::string &name): m_bmrt(bmrt), bmodelPath(name) {
    m_handle = static_cast<bm_handle_t>(bmrt_get_bm_handle(bmrt));
    m_netinfo = bmrt_get_network_info(bmrt, name.c_str());
    assert(m_netinfo->stage_num == 1);

    batchSize = 0;
    if(!m_netinfo->input_num){
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
    if (m_cpu_data != NULL) {
        delete [] m_cpu_data;
        m_cpu_data = NULL;
    }
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

float *BMTensor::get_cpu_data() {
    if (m_cpu_data == NULL) {
        bm_status_t ret;
        float *pFP32 = nullptr;
        int count = bmrt_shape_count(&m_tensor->shape);
        if (m_tensor->dtype == BM_FLOAT32) {
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pFP32, m_tensor->device_mem, count * sizeof(float));
            assert(BM_SUCCESS ==ret);
        }else if (BM_INT8 == m_tensor->dtype) {
            int tensor_size = bmrt_tensor_bytesize(m_tensor);
            int8_t *pU8 = new int8_t[tensor_size];
            assert(pU8 != nullptr);
            pFP32 = new float[count];
            assert(pFP32 != nullptr);
            ret = bm_memcpy_d2s_partial(m_handle, pU8, m_tensor->device_mem, tensor_size);
            assert(BM_SUCCESS ==ret);
            for(int i = 0;i < count; ++ i) {
                pFP32[i] = pU8[i] * m_scale;
            }
            delete [] pU8;
        }else{
            BMLOG(FATAL, "NOT support dtype=%d", m_tensor->dtype);
        }

        m_cpu_data = pFP32;
    }

    return m_cpu_data;
}

}
