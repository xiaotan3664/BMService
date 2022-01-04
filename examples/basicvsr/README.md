BasicVSR Inference
==================

### How to trace models

You have to setup [mmeditting](https://github.com/open-mmlab/mmediting) according to their [guidance](https://mmediting.readthedocs.io/en/latest/install.html#installation) because BasicVSR is published in mmeditting project.

Then apply `export.patch` to export traced models.

```shell
cd path/to/mmeditting
git apply ./export.patch
python3 tools/export_basicvsr.py ./configs/restorers/basicvsr/basicvsr_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth --spy-output spynet.pt --backward-residual-output backward-residual.pt --forward-residual-output forward-residual.pt --upsample-output upsample.pt
```

### FP32 Model Compiling and Inference

To evaluate datasets we have to compile with `dyn` flag to process varying inputs.

```shell
python3 -m bmnetp --model ./spynet.pt --shapes [1,3,160,192],[1,3,160,192] --target BM1684 --outdir spynet.compilation --cmp 0 --dyn True
python3 -m bmnetp --model ./backward-residual.pt --shapes [1,3,144,180],[1,64,144,180] --target BM1684 --outdir backward_residual.compilation --dyn True
python3 -m bmnetp --model ./forward-residual.pt --shapes [1,3,144,180],[1,64,144,180] --target BM1684 --outdir forward_residual.compilation --dyn True
python3 -m bmnetp --model ./upsample.pt --shapes [1,3,144,180],[1,64,144,180],[1,64,144,180] --outdir upsample.compilation --target BM1684 --dyn True
```

To process only 2k inputs, we can compile statically.

```shell
python3 -m bmnetp --model ./spynet.pt --shapes [1,3,544,960],[1,3,544,960] --target BM1684 --outdir spynet.compilation --cmp 0
python3 -m bmnetp --model ./backward-residual.pt --shapes [1,3,144,180],[1,64,144,180] --target BM1684 --outdir backward_residual.compilation
python3 -m bmnetp --model ./forward-residual.pt --shapes [1,3,144,180],[1,64,144,180] --target BM1684 --outdir forward_residual.compilation
python3 -m bmnetp --model upsample.pt --shapes [1,3,144,180],[1,64,144,180],[1,64,144,180] --outdir upsample.compilation --target BM1684
```

Inference and evaluation is straight-forward.

```shell
python3 inference.py --spynet-model ./models/spynet.compilation/compilation.bmodel --forward-residual-model ./models/forward_residual.compilation/compilation.bmodel --backward-residual-model ./models/backward_residual.compilation/compilation.bmodel --upsample-model ./models/upsample.compilation/compilation.bmodel ./data/Vid4/BDx4/calendar/
python3 evaluate.py --spynet-model ./models/spynet.compilation/compilation.bmodel --forward-residual-model ./models/forward_residual.compilation/compilation.bmodel ./models/backward_residual.compilation/compilation.bmodel --upsample-model ./models/upsample.compilation/compilation.bmodel./data/Vid4/BIx4/ ./data/Vid4/GT/
```

You can download Vid4 dataset from [mmeditting](https://mmediting.readthedocs.io/en/latest/sr_datasets.html#vid4-dataset).

### Calibration

In order to do calibration, we have to prepare LMDB datasets for each network. Enable `--dump-input` option when infering fp32 model can easily do that.

```shell
python3 inference.py --spynet-model ./models/spynet.compilation/compilation.bmodel --forward-residual-model ./models/forward_residual.compilation/compilation.bmodel --backward-residual-model ./models/backward_residual.compilation/compilation.bmodel --upsample-model ./models/upsample.compilation/compilation.bmodel ./data/Vid4/BDx4/calendar/ --dump-input
```

Then calibrate each network carefully according to calibration guidance. This repo does not handle input scaling, thus you have to add `--input-as-fp32` and `--output-as-fp32` options when using `bmnetu` to compile int8 bmodel, which makes the models directly compatible.

