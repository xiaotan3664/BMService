from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import sys
import json

def get_image_ids(file_name): 
    ls = [] 
    myset = [] 
    annos = json.load(open(file_name, 'r')) 
    for anno in annos: 
      ls.append(anno['image_id']) 
    myset = {}.fromkeys(ls).keys() 
    myset = sorted(myset)
    return myset 

if __name__ == "__main__":
  det_file = sys.argv[1]
  gt_file = sys.argv[2]
  cocoGt = COCO(gt_file)
  cocoDt = cocoGt.loadRes(det_file)
  image_ids = get_image_ids(det_file)
  cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
  cocoEval.params.imageIds = image_ids
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
