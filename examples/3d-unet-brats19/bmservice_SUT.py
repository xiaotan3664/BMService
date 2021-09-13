import array
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import bmservice

from brats_QSL import get_brats_QSL

from scipy.special import softmax


class BMServiceSUT():
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        print("Loading bmodel...")
        self.runner = bmservice.BMService(model_path)
        # After model conversion output name could be any
        # So we are looking for output with max number of channels

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        print("Finished constructing SUT.")
        self.task_map = {}
        self.map_lock = threading.Lock()

    def wait_result(self):
        while self.left_count>0:
            task_id, values, valid = self.runner.try_get()
            if task_id == 0:
                time.sleep(0.0001)
                continue
            self.map_lock.acquire()
            sample_id = self.task_map[task_id]
            del self.task_map[task_id]
            self.map_lock.release()

            output = values[-1]
            print("got result: task_id={:d} sample_id={:d} shape={}".format(task_id, sample_id, output.shape), flush=True)
            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(sample_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])
            self.left_count -= 1

    def issue_queries(self, query_samples):
        if len(query_samples) == 0:
            return
        self.left_count = len(query_samples)
        print("issue_queries: count=", self.left_count)
        self.receiver = threading.Thread(target=self.wait_result)
        self.receiver.start()
        for i in range(len(query_samples)):
            data = self.qsl.get_features(query_samples[i].index)
            task_id = self.runner.put(data[np.newaxis, ...])
            print("put task_id {:d} sample_id = {:d} with shape = {:}".format(
                task_id, query_samples[i].id, data.shape), flush=True)
            self.map_lock.acquire()
            self.task_map[task_id] = query_samples[i].id
            self.map_lock.release()

    def flush_queries(self):
        print("flush_queries")
        self.receiver.join()
        self.runner.show()
        print("flush_queries finish", flush=True)
                                                  

    def process_latencies(self, latencies_ns):
        pass


def get_bmservice_sut(model_path, preprocessed_data_dir, performance_count):
    return BMServiceSUT(model_path, preprocessed_data_dir, performance_count)
