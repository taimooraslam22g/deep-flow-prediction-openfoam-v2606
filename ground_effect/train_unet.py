"""Train the U-Net of Thuerey et al. on the ground-effect samples, then test it.

The network (unet.py, 2^5 = 32 channels in the first layer), the loss (L1), the optimiser (Adam,
learning rate 0.0006, betas 0.5 and 0.999), the batch size (10), the 80000 iterations and the
learning-rate decay in the second half are those of the reproduction's best run (the original
runTrain.py). Training starts from random weights, as in the original, or from a model trained
on the original free-air airfoils (--pretrained, e.g. the reproduction's long80k_modelG):
transfer learning.

About --val-percent % of the samples in --data (chosen by file name, see
unet_data.is_validation) are kept out of the training to follow it (validation samples), and
--fraction trains on only part of the others, to see how the error depends on the amount of
data. At the end the model is tested on the validation samples and on the --test samples
(airfoils that are not in the training data) with the error measure of the original test
script (unet_data.relative_errors). Results in --out:
    model.pt            the trained weights (unet.TurbNetG(5).load_state_dict(torch.load(...)))
    history.csv         training and validation loss every --eval-every iterations
    errors_<set>.csv    the errors of each validation and test sample, with airfoil, h/c, alpha, U
    summary.json        the settings, the mean errors (also by ride height, angle and speed) and
                        those of the model at the iteration with the lowest validation loss
    train.log           what was printed
With --iterations 0 and --pretrained, the model is only tested (the free-air model as it is).

Example, a short check on the laptop (from the ground_effect folder, Windows):
    ..\\.venv\\Scripts\\python.exe train_unet.py --data data\\dataset\\train --fraction 0.1 --iterations 100 --threads 2 --out data\\unet\\check
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import unet_data
from unet import TurbNetG, weights_init


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder with the training samples (with its subfolders w0, w1, ...)")
    ap.add_argument("--test", help="folder with the test samples (airfoils that are not in --data)")
    ap.add_argument("--out", required=True, help="folder for the results")
    ap.add_argument("--fraction", type=float, default=1.0, help="part of the training samples to use (default 1: all)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed: which samples --fraction takes, the starting weights, the batch order (default 0)")
    ap.add_argument("--iterations", type=int, default=80000, help="training iterations (batches), default 80000")
    ap.add_argument("--pretrained", help="start from this model instead of random weights (e.g. long80k_modelG)")
    ap.add_argument("--lr", type=float, default=0.0006, help="learning rate (default 0.0006)")
    ap.add_argument("--batch", type=int, default=10, help="batch size (default 10)")
    ap.add_argument("--expo", type=int, default=5, help="channel exponent of the network (default 5)")
    ap.add_argument("--val-percent", type=float, default=10, help="percent of the samples kept for validation (default 10)")
    ap.add_argument("--exclude-clipped", action="store_true",
                    help="leave out the samples whose nuTilda was still clipped at the end of the run")
    ap.add_argument("--p-offset", choices=("fluid", "grid"), default="fluid",
                    help="pressure mean removed over the air (default) or over the whole grid (the original)")
    ap.add_argument("--eval-every", type=int, default=1000, help="validation every this many iterations (default 1000)")
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto (default: cuda if there is a GPU)")
    ap.add_argument("--threads", type=int, help="number of CPU threads for torch (default: all)")
    return ap.parse_args(argv)


def compute_lr(i, total, min_lr, max_lr):
    """The original learning-rate schedule (utils.computeLR): max_lr in the first half, then
    halving 6 times towards min_lr. Here per iteration instead of per epoch: the same curve,
    but smooth when an epoch is only a few iterations."""
    if i < total * 0.5:
        return max_lr
    return min_lr + (max_lr - min_lr) * 0.5 ** ((i / float(total) - 0.5) * 2 * 6)


def split(data, val_percent, fraction, seed):
    """The training and the validation samples. For fraction < 1 the training samples are the
    first ones of a random order (set by seed), so a smaller fraction is part of a larger one."""
    val = np.array([unet_data.is_validation(f, val_percent) for f in data["file"]], bool)
    train = np.flatnonzero(~val)
    n = max(1, int(round(fraction * len(train))))
    order = np.random.default_rng(seed).permutation(len(train))
    return unet_data.subset(data, np.sort(train[order[:n]])), unet_data.subset(data, val)


@torch.no_grad()
def predict(net, inputs, batch=50):
    """The model's output (numpy) for inputs, a tensor on the model's device."""
    net.eval()
    return torch.cat([net(inputs[i:i + batch]).cpu() for i in range(0, len(inputs), batch)]).numpy()


def summarise(errors, data):
    """Mean errors of a set of samples, also by ride height, angle of attack and speed."""
    return {"n": len(data["file"]), **{k: float(errors[k].mean()) for k in ("p", "v", "all")},
            "by_h": unet_data.by_bins(errors, data["h"], unet_data.H_EDGES),
            "by_alpha": unet_data.by_bins(errors, data["alpha"], unet_data.ALPHA_EDGES),
            "by_U": unet_data.by_bins(errors, data["U"], unet_data.U_EDGES)}


def write_errors(path, errors, data):
    with open(path, "w") as f:
        f.write("file,airfoil,h_over_c,alpha_deg,U,clipped,error_p,error_v,error_all\n")
        for i, name in enumerate(data["file"]):
            f.write(f"{name},{data['airfoil'][i]},{data['h'][i]:.4f},{data['alpha'][i]:.4f},{data['U'][i]:.3f},"
                    f"{int(data['clipped'][i])},{errors['p'][i]:.6f},{errors['v'][i]:.6f},{errors['all'][i]:.6f}\n")


def run(args, data=None, test=None):
    """Train and test one model (see the top). data and test are dicts from unet_data.load, so
    that learning_curve.py loads the samples only once; None loads them from args.data, args.test."""
    os.makedirs(args.out, exist_ok=True)
    log_file = open(os.path.join(args.out, "train.log"), "w")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    if args.iterations == 0 and not args.pretrained:
        raise SystemExit("--iterations 0 only tests a model, so it needs --pretrained")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # the input size never changes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()
    if data is None:
        data = unet_data.load(args.data, args.p_offset, args.exclude_clipped)
    if test is None and args.test:
        test = unet_data.load(args.test, args.p_offset, args.exclude_clipped)
    train, val = split(data, args.val_percent, args.fraction, args.seed)
    log(f"{len(train['file'])} training samples ({args.fraction:g} of {len(data['file']) - len(val['file'])}), "
        f"{len(val['file'])} validation, {len(test['file']) if test else 0} test; device {device}; "
        f"loaded in {time.time() - start:.0f} s")
    if args.iterations and len(train["file"]) < args.batch:
        raise SystemExit(f"fewer training samples than the batch size ({args.batch})")

    net = TurbNetG(channelExponent=args.expo).to(device)
    net.apply(weights_init)
    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))
        log(f"starting from {args.pretrained}")
    log(f"{sum(p.numel() for p in net.parameters() if p.requires_grad)} trainable parameters")

    to_device = lambda a: torch.from_numpy(a).to(device)
    X, Y, X_val = to_device(train["inputs"]), to_device(train["targets"]), to_device(val["inputs"])
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0)
    criterion = nn.L1Loss()
    order = torch.Generator().manual_seed(args.seed)
    history = open(os.path.join(args.out, "history.csv"), "w")
    history.write("iteration,lr,train_l1,val_l1,val_error_p,val_error_v,val_error_all,seconds\n")
    best = (np.inf, 0, None)  # validation loss, iteration, weights
    it, loss_sum, loss_n, t0 = 0, torch.zeros((), device=device), 0, time.time()
    net.train()
    while it < args.iterations:
        perm = torch.randperm(len(X), generator=order)
        for b in range(len(X) // args.batch):  # incomplete last batch dropped, as in the original
            idx = perm[b * args.batch:(b + 1) * args.batch].to(device)
            lr = compute_lr(it, args.iterations, 0.1 * args.lr, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(X[idx]), Y[idx])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            loss_n += 1
            it += 1
            if it % args.eval_every == 0 or it == args.iterations:
                pred = predict(net, X_val)
                err = unet_data.relative_errors(pred, val["targets"])
                val_l1 = float(np.abs(pred - val["targets"]).mean())
                train_l1 = float(loss_sum) / loss_n
                loss_sum.zero_()
                loss_n = 0
                if val_l1 < best[0]:
                    best = (val_l1, it, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                took = time.time() - t0
                history.write(f"{it},{lr:.6g},{train_l1:.6g},{val_l1:.6g},{err['p'].mean():.6g},"
                              f"{err['v'].mean():.6g},{err['all'].mean():.6g},{took:.1f}\n")
                history.flush()
                log(f"{it:6d}/{args.iterations}  lr {lr:.1e}  train L1 {train_l1:.5f}  val L1 {val_l1:.5f}  "
                    f"val error {100 * err['all'].mean():5.2f}%  ({took / 60:.1f} min, about "
                    f"{took / it * (args.iterations - it) / 60:.0f} min left)")
                net.train()
            if it == args.iterations:
                break
    history.close()
    train_seconds = time.time() - t0
    torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))

    summary = {"settings": vars(args), "n_train": len(train["file"]), "device": str(device),
               "torch": torch.__version__, "train_seconds": round(train_seconds, 1)}
    sets = [("val", val)] + ([("test", test)] if test else [])
    for name, d in sets:
        err = unet_data.relative_errors(predict(net, to_device(d["inputs"])), d["targets"])
        write_errors(os.path.join(args.out, f"errors_{name}.csv"), err, d)
        summary[name] = summarise(err, d)
        log(f"{name}: {summary[name]['n']} samples, error p {100 * summary[name]['p']:.2f}%, "
            f"v {100 * summary[name]['v']:.2f}%, all {100 * summary[name]['all']:.2f}%")
    if best[2] is not None:  # the model at the lowest validation loss, for comparison
        net.load_state_dict(best[2])
        summary["best_val"] = {"iteration": best[1]}
        for name, d in sets:
            err = unet_data.relative_errors(predict(net, to_device(d["inputs"])), d["targets"])
            summary["best_val"][name] = {k: float(err[k].mean()) for k in ("p", "v", "all")}
        log(f"lowest validation loss at iteration {best[1]}: " + ", ".join(
            f"{name} all {100 * summary['best_val'][name]['all']:.2f}%" for name, _ in sets))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"done in {(time.time() - start) / 60:.1f} min, results in {args.out}")
    log_file.close()
    return summary


def main(argv=None):
    run(parse(argv))


if __name__ == "__main__":
    main()
