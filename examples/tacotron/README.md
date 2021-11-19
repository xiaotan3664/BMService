Tacotron2 & Waveglow Inference
==============================

### Prerequisites

```
apt-get install llvm-8 libsndfile1-dev
ln -s /usr/bin/llvm-config-8 /usr/bin/llvm-config
pip3 install --upgrade setuptools
pip3 install nvidia-pyindex
pip3 install -r requirements.txt
```

Download model checkpoints from [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2). FYI, surely compatible models' md5sums are here.

```
b8a9f0f782ab04731d81af27182b29e9  tacotron2_1032590_6000_amp
596c69f1ab117020bcbe47ab344c6e52  waveglow_1076430_14000_amp
```

### Compile models

```
python3 build_bmodel.py --tacotron checkpoints/tacotron2_1032590_6000_amp --waveglow checkpoints/waveglow_1076430_14000_amp all
```

The onnx models will be stored in `onnx_models/` dir, bmodels reside in seperate compilation dirs.

### Inference

```
python3 inference_bmservice.py --threads 1 -i phrases/phrase.txt -o out
```

In order to fully utilize hardware resource, `--threads` should be at least the same with TPU core number.

