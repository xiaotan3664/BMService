import os
import ctypes as ct
import numpy as np

BMTypeTuple = (
   (np.float32, 0),
   (np.int32, 6),
   (np.uint32, 7),
   (np.int8, 2),
   (np.uint8, 3),
)
def bmlen(t):
    if t in [2,3]:
        return 1
    elif t in [0,6,7]:
        return 4

def bmtype(t):
    for nt, bt in BMTypeTuple: 
        if t == nt:
            return bt
def nptype(t):
    for nt, bt in BMTypeTuple: 
        if t == bt:
            return nt

class BMTensor(ct.Structure):
    _fields_ = [
        ("dims", ct.c_uint32),
        ("shape", ct.c_uint32*8),
        ("dtype", ct.c_uint32),
        ("data", ct.c_void_p),
    ]
    def to_numpy(self):
        shape = self.shape[0:self.dims]
        dtype = nptype(self.dtype)
        mem_size = np.prod(shape)*bmlen(self.dtype)
        data_ptr = (ct.c_byte*mem_size)()
        ct.memmove(data_ptr, self.data, mem_size)
        #data_ptr = ct.cast(self.data, ct.POINTER(ct.c_byte*mem_size)).contents
        buffer = data_ptr
        return np.frombuffer(buffer, dtype = dtype).reshape(shape)
        
    def from_numpy(self, data):
        self.dims = ct.c_uint32(len(data.shape))
        for i in range(self.dims):
            self.shape[i] = ct.c_uint32(data.shape[i])
        self.dtype = ct.c_uint32(bmtype(data.dtype))
        self.data = data.ctypes.data_as(ct.c_void_p)

class BMService:
    def __init__(self, bmodel_path):
        self.bmodel_path = bmodel_path
        self.lib_path = os.path.join(os.path.dirname(__file__), "lib/libbmservice.so")
        self.__lib = ct.cdll.LoadLibrary(self.lib_path)
        self.runner_id = self.__lib.runner_start(ct.c_char_p(bytes(bmodel_path, encoding='utf-8')))

    def __del__(self):
        self.__lib.runner_stop(self.runner_id)

    def put(self, *inputs):
        input_num = ct.c_int(len(inputs))
        bm_inputs = (BMTensor*len(inputs))()
        for i in range(len(inputs)):
            bm_inputs[i].from_numpy(inputs[i])
        task_id = self.__lib.runner_put_input(self.runner_id, input_num, bm_inputs, 1)
        return task_id
        
    def get(self):
        return self.__get(self.__lib.runner_get_output)

    def try_get(self):
        return self.__get(self.__lib.runner_try_to_get_output)

    def stopped(self):
        return self.__lib.runner_all_stopped(self.runner_id)
        
    def empty(self):
        return self.__lib.runner_empty(self.runner_id)

    def __get(self, func):
        output_num= ct.c_uint32(0)
        task_id = ct.c_uint32(0)
        output_valid = ct.c_uint32(0)
        func.restype = ct.POINTER(BMTensor)
        output_tensors = func(self.runner_id, ct.byref(task_id), ct.byref(output_num), ct.byref(output_valid))
        if(task_id.value == 0):
            return 0, [], 0
        outputs = []
        if output_valid.value == 0:
            return task_id, [], False
        for i in range(output_num.value):
            outputs.append(output_tensors[i].to_numpy())
        self.__lib.runner_release_output(output_num, output_tensors)
        return task_id.value, outputs, True

    def inference(self, *inputs):
        in_task_id = self.put(*inputs)
        out_task_id, outputs, valid = self.get()
        assert in_task_id == out_task_id
        return outputs, valid


    def show(self):
        self.__lib.runner_show_status(self.runner_id)


if __name__ == "__main__":

    n = np.arange(1*3*2*2).astype(np.float32).reshape(1,3,2,2)
    t = BMTensor()
    t.from_numpy(n)
    nn = t.to_numpy()
    n[0,0,0,0]=2.4
    print(n)
    print(nn)
    bmodel_path = os.path.join(os.path.dirname(__file__), "test_model/compilation.bmodel")
    s = BMService(bmodel_path)

    i = np.arange(1*3*20*20).astype(np.float32).reshape(1,3,20,20)
    print(i)
    print(s.inference(i))
