import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import nullcontext
from tqdm import tqdm
from collections import defaultdict
from open_clip import tokenizer

class AverageMeter(object):
    def __init__(self):
        self.num = 0
        self.tot = 0

    def update(self, val, sz):
        self.num += val*sz
        self.tot += sz

    def mean(self):
        if self.tot==0: return None
        return self.num/self.tot

def get_margins_given_logits(L, Y, clone_data=True):
    """
    Args
    - L: logits
    - Y: labels
    """
    L = L.clone() if clone_data else L
    rng = torch.arange(len(L))
    class_logits = L[rng, Y]
    L[rng, Y] = -np.inf
    max_wo_class = L[rng, L.argmax(1)]
    M = class_logits-max_wo_class
    return M

def get_accuracy_and_loss(model, loader, device, loss_fn=F.cross_entropy, use_tqdm=True,
                          enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    # loss_fn: any function that takes in (logits, targets) and outputs a scalar
    # assert next(model.parameters()).device == device

    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        if use_tqdm:
            loader = tqdm(loader, desc='Accuracy')
        for xb, yb, *_ in loader:
            bs = len(xb)
            xb, yb = xb.to(device), yb.to(device)

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)

                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            preds = out.argmax(-1)

            b_acc = (preds==yb).float().mean().item()
            b_loss = loss_fn(out, yb).item()

            acc_meter.update(b_acc, bs)
            loss_meter.update(b_loss, bs)

            xb, yb = xb.cpu(), yb.cpu()
            if use_tqdm:
                loader.set_postfix({'Acc': f'{acc_meter.mean():.2f}'})

    if in_tr_mode:
        model.train()

    return {
        'acc': acc_meter.mean(),
        'loss': loss_meter.mean()
    }

def get_confusion_matrix(model, loader, num_classes, device,
                         enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    """[label][pred] = count"""
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()

    cmat = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device)

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)
                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            yb = yb.cpu().clone()
            preds = out.argmax(-1).cpu().clone()

            for y, yh in zip(yb.numpy(), preds.numpy()):
                cmat[y][yh] += 1

            xb = xb.cpu()

    if in_tr_mode:
        model.train()

    return cmat

def get_classwise_accuracies(model, loader, num_classes, device, lock=None, use_eval=True, lr_tta=False):
    cmat = get_confusion_matrix(model, loader, num_classes, device, lock=lock, use_eval=use_eval, lr_tta=lr_tta)
    return np.diag(cmat)/cmat.sum(axis=1)

def get_predictions(model, loader, device, enable_amp=True, lock=None, use_eval=True, lr_tta=False):
    # assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device)

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)
                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            yh = out.argmax(-1).cpu().numpy()
            preds.append(yh)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

            xb = xb.cpu()

    if in_tr_mode:
        model.train()

    preds, labels = map(np.concatenate, [preds, labels])
    return preds, labels

def get_labels(loader, label_index=1):
    labels = []
    for tb in loader:
        yb = tb[label_index]
        yb = yb.cpu().numpy()
        labels.append(yb)
    labels = np.concatenate(labels)
    return labels

def get_margins(model, loader, device, enable_amp=True, lock=None, use_tqdm=False, use_eval=True, lr_tta=False):
    # assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    all_margins = []
    lock = lock if lock is not None else nullcontext()

    with torch.no_grad():
        pbar = tqdm(loader) if use_tqdm else loader
        for xb, yb, *_ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.long()
            rng = torch.arange(len(xb))

            with autocast(enabled=enable_amp):
                with lock:
                    out = model(xb)
                    if lr_tta:
                        out += model(torch.fliplr(xb))
                        out /= 2

            class_logits = out[rng, yb].clone()
            out[rng, yb] = -np.inf
            max_wo_class = out[rng, out.argmax(1)]
            class_logits = (class_logits - max_wo_class).cpu()
            all_margins.append(class_logits)

    if in_tr_mode:
        model = model.train()

    all_margins = torch.cat(all_margins).numpy()
    return all_margins

def get_margins_batch(model, xb, yb, device, enable_amp=True, use_eval=True, **kw):
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    all_margins = []

    with torch.no_grad():
        xb = xb.to(device, non_blocking=True)
        yb = yb.long()
        rng = torch.arange(len(xb))

        with autocast(enabled=enable_amp):
            out = model(xb)

        class_logits = out[rng, yb].clone()
        out[rng, yb] = -np.inf
        max_wo_class = out[rng, out.argmax(1)]
        class_logits = (class_logits - max_wo_class).cpu()
        all_margins.append(class_logits)

    if in_tr_mode:
        model = model.train()

    all_margins = torch.cat(all_margins).numpy()
    return all_margins

def get_logits(model, loader, device, enable_amp=True, lock=None,
               apply_fn=None, use_eval=True, lr_tta=False, with_labels=False):
    # assert next(model.parameters()).device == device
    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()
    all_logits = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            if apply_fn: xb = apply_fn(xb)
            xb = xb.to(device, non_blocking=True)

            with autocast(enabled=enable_amp):
                with lock:
                    logits = model(xb)
                    if lr_tta:
                        logits += model(torch.fliplr(xb))
                        logits /= 2
                    logits = logits.clone().cpu()
            all_logits.append(logits)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

    if in_tr_mode:
        model = model.train()

    all_logits = torch.cat(all_logits).numpy()
    labels = np.concatenate(labels)

    if with_labels: return all_logits, labels
    return all_logits

def get_loss_vals(model, loader, device, loss_fn=F.cross_entropy, enable_amp=True,
                  lock=None, apply_fn=None, use_eval=True, lr_tta=False,
                  with_labels=False, **loss_kwargs):
    # assert next(model.parameters()).device == device
    kw = {'reduce': False, 'reduction': 'none'}
    kw.update(loss_kwargs)

    in_tr_mode = model.training
    if use_eval: model = model.eval()
    lock = lock if lock is not None else nullcontext()
    vals = []
    labels = []

    with torch.no_grad():
        for xb, yb, *_ in loader:
            if apply_fn:
                xb = apply_fn(xb)
            yb = yb.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)

            with autocast(enabled=enable_amp):
                with lock:
                    logits = model(xb)
                    if lr_tta:
                        logits += model(torch.fliplr(xb))
                        logits /= 2

            b_loss = loss_fn(logits, yb, **kw).detach().clone()
            vals.append(b_loss)

            yb = yb.clone()
            labels.append(yb)

    if in_tr_mode:
        model = model.train()

    vals = torch.concat(vals).cpu().numpy()
    labels = torch.concat(labels).cpu().numpy()

    if with_labels: return vals, labels
    return vals

def get_residuals(model, loader, device, enable_amp=True, lock=None, apply_fn=None):
    logits, labels = get_logits(model, loader, device,
                                enable_amp=enable_amp,
                                lock=lock,
                                with_labels=True,
                                apply_fn=apply_fn)

    logits = logits.reshape(-1)
    return (logits-labels)**2

def get_groupwise_accuracy(model, loader, device, group_index=2, add_class_to_group=True, lock=None):
    in_tr_mode = model.training
    model = model.eval()
    lock = lock if lock is not None else nullcontext()
    wg_meters = defaultdict(AverageMeter)

    with torch.no_grad():
        for tup in loader:
            xb, yb = tup[:2]
            if group_index >= len(tup):
                assert False, "group_index is too large"
            mb = tup[group_index]
            xb = xb.to(device, non_blocking=True)
            yb, mb = map(lambda z: z.clone().cpu().numpy(), [yb, mb])

            with autocast(enabled=True):
                with lock:
                    logits = model(xb)

            preds = logits.argmax(-1).cpu().numpy()
            is_correct = (preds==yb).astype(int)

            for is_c, y, m in zip(is_correct, yb, mb):
                if add_class_to_group:
                    wg_meters[(y,m)].update(is_c, 1)
                else:
                    wg_meters[m].update(is_c, 1)

    if in_tr_mode:
        model = model.train()

    group_accs = {k:v.mean() for k,v in wg_meters.items()}
    return group_accs

def get_clip_margins(model, loader, class_embeddings, *,
                     inv_temperature=100.0, use_tqdm=False,
                     normalize_image_features=True,
                     device=torch.device(0), eval_margins_on_device=True):
    # use clipmodel_logits function
    logits, labels = get_clipmodel_logits(model, loader, class_embeddings,
                                          inv_temperature=inv_temperature,
                                          use_tqdm=use_tqdm,
                                          device=device,
                                          with_labels=True,
                                          normalize_image_features=normalize_image_features)

    # use margins_given_logits function
    margins_device = device if eval_margins_on_device else torch.device('cpu')
    logits = torch.FloatTensor(logits).to(margins_device)
    labels = torch.LongTensor(labels).to(margins_device)
    margins = get_margins_given_logits(logits, labels)
    margins = margins.cpu().numpy()
    return margins

def get_clip_logits(model, loader, class_embeddings, *,
                    inv_temperature=100.0, use_tqdm=False,
                    normalize_image_features=True,
                    device=torch.device(0), with_labels=False):
    """
    args
    - model: clip model from open_clip
    - loader: dataloader
    - class_embeddings: embeddings of classes [num_classes, embedding_dim]
    - inv_temperature: inverse temperature for scaling logits
    - use_tqdm: whether to use tqdm
    - device: device to run on
    - with_labels: whether to return labels
    """
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    model = model.eval()

    all_logits = []
    labels = []

    class_embeddings = class_embeddings.to(device)
    zs_weights = class_embeddings.T.half()


    loader = tqdm(loader) if use_tqdm else loader

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device, non_blocking=True)

            with autocast(enabled=True):
                image_features = model.encode_image(xb)

            if normalize_image_features:
                image_features /= image_features.norm(dim=-1, keepdim=True)

            logits= inv_temperature * (image_features @ zs_weights)
            logits = logits.clone().cpu()
            all_logits.append(logits)

            yb = yb.clone().cpu().numpy()
            labels.append(yb)

    if in_tr_mode:
        model = model.train()

    all_logits = torch.cat(all_logits).numpy()
    labels = np.concatenate(labels)

    if with_labels: return all_logits, labels
    return all_logits

def get_clip_features(model, loader, *,
                      use_tqdm=True,
                      normalize_image_features=True,
                      device=torch.device(0),
                      enable_amp=True):
    """
    args
    - model: clip model from open_clip
    - loader: dataloader
    - use_tqdm: whether to use tqdm
    - device: device to run on
    """
    assert next(model.parameters()).device == device
    in_tr_mode = model.training
    model = model.eval()

    all_logits = [] # image features

    loader = tqdm(loader) if use_tqdm else loader

    with torch.no_grad():
        for xb, yb, *_ in loader:
            xb = xb.to(device, non_blocking=True).float()

            with autocast(enabled=enable_amp):
                image_features = model.encode_image(xb)

            if normalize_image_features:
                image_features /= image_features.norm(dim=-1, keepdim=True)

            all_logits.append(image_features.clone().cpu())

    if in_tr_mode:
        model = model.train()

    all_logits = torch.cat(all_logits).numpy()
    return all_logits

def get_clip_label_embeddings(clip_model, classnames, templates, use_tqdm=False):
    """
    Args
    - clip_model: clip model from open_clip
    - classnames: list of classnames
    - templates: list of templates to use for each classname

    returns [classes x embed_dim] matrix, each embedding is normalized
    taken from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    """
    with torch.no_grad():
        zeroshot_weights = []
        itr = tqdm(classnames) if use_tqdm else classnames
        for classname in itr:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer.tokenize(texts).cuda() #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.T