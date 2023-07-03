import jukemirlib
import torch


def extractor(fpath):
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    audio = jukemirlib.load_audio(fpath)
    reps = jukemirlib.extract(audio, layers=[36])
    return torch.tensor(reps[36])
    