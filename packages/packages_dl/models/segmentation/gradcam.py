import torch
import torch.nn.functional as F

def gsp2d(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out

def gap2d_pos(x, keepdims=False):
    out = torch.sum(x.view(x.size(0), x.size(1), -1), -1) / (torch.sum(x>0)+1e-12)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def gap2d(x, keepdims=False):
  out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
  if keepdims:
      out = out.view(out.size(0), out.size(1), 1, 1)

  return out

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits_vec).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image, step=1, get_prob=False):
        # self.image_shape = image.shape[2:]
        self.logits = self.model(image, step=step)
        self.logits_vec = F.relu(self.logits)
        self.logits_vec = gsp2d(self.logits_vec, keepdims=True)[:, :, 0, 0]
        self.probs = F.softmax(self.logits, dim=1)
        if get_prob:
            return F.softmax(self.logits_vec, dim=1)
        else:
            return self.logits  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        # one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits_vec[:, ids].sum().backward(retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

class CAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(CAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0]

            return backward_hook


        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                # print(name)
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))
                # print(self.fmap_pool)
                # print(self.grad_pool)
        # print(self.candidate_layers)
                
    def _find(self, pool, target_layer):
        # print(pool.keys())
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        # print(self.fmap_pool)
        # print(self.grad_pool)
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)

        weights = gap2d_pos(grads, keepdims=True)

        gcam = torch.mul(fmaps, weights)


        gcam = gcam.sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        return gcam