import os
import sys
import numpy as np
from run_squad import (read_squad_examples, convert_examples_to_features,
                       tokenization, write_predictions, RawResult)
import bmservice
import threading
import time


bmodel_path = "/home/longli/tingliangtan/BMService/models/bert_squad/fp32_b8.bmodel"
vocab_file = "uncased_L-12_H-768_A-12/vocab.txt"
input_file = "squad/dev-v1.1.json"
output_dir = "squad_eval_out"
max_seq_length = 384
doc_stride = 128
max_query_length = 64 
do_lower_case = True
n_best_size = 20
max_answer_length=30
max_batch = 8

def main():
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
  eval_examples = read_squad_examples(input_file, False)
  eval_features = []
  id_maps = {}
  id_maps_lock = threading.Lock()
  runner = bmservice.BMService(bmodel_path)
  finished = False
  
  eval_examples = eval_examples[0:3]
  batch_input_ids = []
  batch_input_mask = []
  batch_segment_ids = []
  batch_unique_id = []
  def feed_feature(feature):
    nonlocal batch_unique_id
    nonlocal batch_input_ids
    nonlocal batch_input_mask
    nonlocal batch_segment_ids
    eval_features.append(feature)
    batch_unique_id.append(feature.unique_id)
    batch_input_ids += feature.input_ids
    batch_segment_ids += feature.segment_ids
    batch_input_mask += feature.input_mask
    if len(batch_unique_id) < max_batch:
      return
    input_ids_data = np.array(batch_input_ids, dtype=np.int32).reshape(-1, max_seq_length)
    segment_ids_data = np.array(batch_segment_ids, dtype=np.int32).reshape(-1, max_seq_length)
    input_mask_data = np.array(batch_input_mask, dtype=np.int32).reshape(-1, max_seq_length)
    task_id = runner.put(input_ids_data, segment_ids_data, input_mask_data)
    id_maps_lock.acquire()
    id_maps[task_id] = [] + batch_unique_id
    print("put", task_id, batch_unique_id)
    id_maps_lock.release()
    batch_unique_id.clear()
    batch_segment_ids.clear()
    batch_input_ids.clear()
    batch_input_mask.clear()

  def convert_features_thread(eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length, feed_feature):
    nonlocal batch_unique_id
    nonlocal batch_input_ids
    nonlocal batch_input_mask
    nonlocal batch_segment_ids
    convert_examples_to_features(eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length, False, feed_feature)
    if len(batch_input_ids) != 0:
      input_ids_data = np.array(batch_input_ids, dtype=np.int32).reshape(-1, max_seq_length)
      segment_ids_data = np.array(batch_segment_ids, dtype=np.int32).reshape(-1, max_seq_length)
      input_mask_data = np.array(batch_input_mask, dtype=np.int32).reshape(-1, max_seq_length)
      task_id = runner.put(input_ids_data, segment_ids_data, input_mask_data)
      id_maps_lock.acquire()
      id_maps[task_id] = []+batch_unique_id
      print("put", task_id, batch_unique_id)
      id_maps_lock.release()
      batch_unique_id.clear()
      batch_segment_ids.clear()
      batch_input_ids.clear()
      batch_input_mask.clear()
    while not runner.stopped():
      runner.put()
      time.sleep(0.0001)

  input_thread = threading.Thread(target=convert_features_thread,
                                  args=(eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length, feed_feature))
  input_thread.start()

  eval_results = []
  while not (runner.stopped() and runner.empty()):
    task_id, values, valid = runner.try_get()
    if task_id == 0:
      time.sleep(0.0001)
      continue
    id_maps_lock.acquire()
    unique_ids = id_maps[task_id]
    id_maps_lock.release()
    print("got", task_id, unique_ids)
    start_logits, end_logits = values
    batch = len(unique_ids)
    for b in range(batch):
      unique_id = unique_ids[b]
      start_logit = [float(x) for x in start_logits[b,:].flat]
      end_logit = [float(x) for x in end_logits[b,:].flat]
      eval_results.append(RawResult(unique_id, start_logit, end_logit))
  print(runner.stopped(), runner.empty())

  runner.show()
  input_thread.join()

  output_prediction_file = os.path.join(output_dir, "predictions.json")
  output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
  output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")

  write_predictions(eval_examples, eval_features, eval_results, n_best_size,
                        max_answer_length, do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file)

if __name__ == "__main__":
  main()
