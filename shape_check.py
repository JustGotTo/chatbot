import torch


class ShapeChecker:
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.handles = []

    def _hook(self, name):
        def fn(module, inp, out):
            if isinstance(inp, tuple):
                inp_shapes = [x.shape for x in inp if isinstance(x, torch.Tensor)]
            else:
                inp_shapes = [inp.shape]

            if isinstance(out, tuple):
                out_shapes = [x.shape for x in out if isinstance(x, torch.Tensor)]
            else:
                out_shapes = [out.shape]

            if self.verbose:
                print(f"[{name}]")
                print(f"   Input : {inp_shapes}")
                print(f"   Output: {out_shapes}\n")

        return fn

    def enable(self):
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in [
                "embedding", "encoder", "decoder",
                "attention", "linear", "layer_norm",
                "feedforward"
            ]):
                h = module.register_forward_hook(self._hook(name))
                self.handles.append(h)

    def disable(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()