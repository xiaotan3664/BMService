import torch
import numpy as np
from scipy.io.wavfile import write

def save_audios(infer_fn):
    def fun(fn, mel, *args):
        batch_size, _, mel_size = mel.shape
        stride = 256
        n_group = 8
        stft_hop_length = 256
        sample_rate = 22050
        z_size = mel_size*stride
        z_size = z_size//n_group
        z = torch.randn(batch_size, n_group, z_size).numpy()
        audio = infer_fn(mel, z, *args)[0]
        audio = torch.from_numpy(audio)
        audio = audio[:mel_size*stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        write(fn, sample_rate, audio.cpu().numpy())
    return fun

@save_audios
def onnx_infer(mel, z, model_path):
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = dict(mel=mel, z=z)
    return ort_session.run(None, ort_inputs)[0]

@save_audios
def bm_infer(mel, z, model_path):
    import bmservice
    rt = bmservice.BMService(model_path)
    rt.put(mel, z)
    task_id, values, ok = rt.get()
    if not ok:
        raise RuntimeException('bm rt not ok')
    return values[0]

def main():
    import sys
    if len(sys.argv) != 4:
        print(sys.argv[0], '<mel.npy> <.onnx> <.bmodel>')
        return
    mel_fn, onnx_model, bmodel = sys.argv[1:]
    mel = np.load(mel_fn)
    onnx_infer('onnx.wav', mel, onnx_model)
    bm_infer('bm.wav', mel, bmodel)

if __name__ == '__main__':
    main()
