import io
import torch
import torch.nn as nn

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # TODO: serialize src's state dict into a buffer, rewind, then load it into dst

    # create buffer
    buf = io.BytesIO()

    torch.save(src.state_dict(), buf)
    buf.seek(0)
    model_state = dst.load_state_dict(torch.load(buf))
    # torch.save(model_state, dst)