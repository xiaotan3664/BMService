import sys
import numpy as np
import sklearn.metrics
import threading
from os import path
import time

class DLRMRunner:
  def __init__(self, bmodel_path):
    self.bmodel_path = bmodel_path
    self.batch_num = 1024
    self.print_interval = 10

  def run(self, data_path):
    self.load_data(data_path)
    self.left_count = self.num_samples
    import bmservice
    self.runner = bmservice.BMService(self.bmodel_path)
    self.y_score = np.zeros_like(self.y_true)

    self.task_map = {}
    self.map_lock = threading.Lock()
    feed_thread = threading.Thread(target=self.feed_data)
    feed_thread.start()

    while self.left_count>0:
      task_id, values, valid = self.runner.try_get()
      if task_id == 0:
          time.sleep(0.0001)
          continue
      self.map_lock.acquire()
      start_index, end_index = self.task_map[task_id]
      del self.task_map[task_id]
      self.map_lock.release()
      if task_id % self.print_interval == 0:
        print("get task_id={}, start={}, end={}".format(task_id, start_index, end_index), flush=True)
      output = values[0]
      out_num = end_index - start_index
      self.y_score[start_index:end_index] = output[0:out_num].reshape(-1)
      self.left_count = self.left_count - out_num

    self.runner.show()
    data_dir = path.dirname(data_path)
    y_score_file = path.join(data_dir, "dlrm_test_y_score.npy")
    print("save y_score data to {}, which may take minutes...".format(y_score_file))
    np.save(y_score_file, self.y_score)
    #print("y=", self.y_score)
    return self.y_score, self.y_true

  def load_data(self, data_path):
    data_dir = path.dirname(data_path)
    print("loading data from {}, which may take minutes...".format(data_path))
    int_fea_file = path.join(data_dir, "dlrm_test_int_fea.npy")
    cat_fea_file = path.join(data_dir, "dlrm_test_cat_fea.npy")
    y_file = path.join(data_dir, "dlrm_test_y.npy")
    if path.exists(int_fea_file) and path.exists(cat_fea_file) and path.exists(y_file):
      self.x_int = np.load(int_fea_file)
      self.x_cat = np.load(cat_fea_file)
      self.y_true = np.load(y_file)
      self.num_samples = len(self.y_true)
      print("find cached sample files, load directly!")
      return
    test_data = np.load(data_path)
    x_int, x_cat, y_true = test_data["X_int"], test_data["X_cat"], test_data["y"]
    self.num_samples = len(y_true)//2;
    print("prepare samples: sample_num={}".format(self.num_samples))
    self.x_int = x_int[0:self.num_samples].astype(np.float32)
    self.x_cat = x_cat[0:self.num_samples].astype(np.int32)
    self.y_true = y_true[0:self.num_samples].astype(np.float32)
    np.save(int_fea_file, self.x_int)
    np.save(cat_fea_file, self.x_cat)
    np.save(y_file, self.y_true)
    print("prepare done!")

  def feed_data(self):
    num_cat = self.x_cat[0].shape[0]
    num_int = self.x_int[0].shape[0]
    batch_num = self.batch_num
    offset_batch = np.tile(np.arange(batch_num, dtype=np.int32).reshape(1,-1), (num_cat, 1))
    offset_batch = np.ascontiguousarray(offset_batch)
    feed_count = 0
    while feed_count < self.num_samples:
      left_count = self.num_samples - feed_count
      start_index = feed_count
      if left_count<batch_num:
        end_index = self.num_samples
        int_batch = np.zeros((batch_num, num_int), dtype=np.float32)
        cat_batch = np.zeros((batch_num, num_cat), dtype=np.int32)
        int_batch[0:end_index - start_index] = np.log(1+self.x_int[start_index:end_index])
        cat_batch[0:end_index - start_index] = self.x_cat[start_index:end_index]
      else:
        end_index = feed_count + batch_num
        int_batch = np.log(1+self.x_int[start_index: end_index])
        cat_batch = self.x_cat[start_index: end_index]
      feed_count += end_index - start_index
      int_batch = np.ascontiguousarray(int_batch)
      cat_batch_splitted = [ np.ascontiguousarray(c) for c in np.split(cat_batch, num_cat, -1) ]
      task_id = self.runner.put(int_batch, offset_batch, *cat_batch_splitted) 
      self.map_lock.acquire()
      self.task_map[task_id] = (start_index, end_index)
      self.map_lock.release()
      if task_id % self.print_interval==0:
        print("put task_id={}, start={}, end={}".format(task_id, start_index, end_index), flush=True)

if __name__ == "__main__":
   data_path = sys.argv[1]
   bmodel_path = sys.argv[2]
   runner = DLRMRunner(bmodel_path)
   y_score, y_true = runner.run(data_path)
   print("start to calculate AUC score...")
   auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)
   print("---> auc_score = {}".format(auc_score))
