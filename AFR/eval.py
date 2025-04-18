import os
import torch
from torch.utils.data import DataLoader
from dataset import VADDataset, create_dataloaders
from utils import *
from model import VADTransformer
from tqdm import tqdm
from sklearn.metrics import (classification_report, auc, roc_curve,
                             precision_recall_curve, accuracy_score)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast
import warnings
from sklearn.exceptions import UndefinedMetricWarning

plt.switch_backend("agg")

def plot_result(save_root, vid, gt, pred):
    n_layers = len(pred)
    colors = [plt.cm.tab20(i) for i in range(20)]

    x = np.arange(len(gt))
    plt.fill_between(x, gt, where=gt > 0, facecolor="red", alpha=0.2)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("Frame number")
    plt.ylabel("Anomaly scores")

    for i in range(n_layers):
        plt.plot(x, pred[i], color=colors[i], label=f"fpn_level_{i}", linewidth=2)

    pred_t = np.stack(pred).mean(axis=0)
    plt.plot(x, pred_t, color="r", label="final_score", linewidth=3)

    plt.grid(True, linestyle="-.")
    plt.legend()
    plt.savefig(f"{save_root}/{vid}.png", dpi=100, bbox_inches="tight")
    plt.clf()

@torch.no_grad()
def eval(args, plot=False):
    test_loader = DataLoader(
        VADDataset(args, test_mode=True),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VADTransformer(args)
    restore_checkpoint(args, model, args.ckpt_file)
    if args.device == "cuda":
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    model.eval()

    print(f"Start evaluating {args.ckpt_file} ...")
    eval_results = eval_one_step(args, test_loader, model, threshold=0.5, plot=plot)
    eval_str = "".join(f" {k}:{v:2.4} " for k, v in eval_results.items() if k not in ["frame_level_report", "video_level_report"])
    print(f"Model: {args.ckpt_file}, {eval_str}")
    print(f"Frame-level classification report:\n{eval_results['frame_level_report']}")
    print(f"Video-level classification report:\n{eval_results['video_level_report']}")

@torch.no_grad()
def eval_one_step(args, val_loader, model, threshold=0.5, plot=False):
    model.eval()
    vids, predictions, num_frames = [], [], []

    for batch in val_loader:
        inputs = {k: batch[k].to(args.device) for k in ["feats", "masks"]}
        with autocast(enabled=args.amp):
            scores = model(inputs, is_training=False)

        scores = [
            upgrade_resolution_th(scores[i], scale=args.scale_factor**i * args.window_size, mode="linear")
            for i in range(len(scores))
        ]
        pred = torch.stack(scores, dim=-1).mean(dim=-1).cpu().numpy()
        predictions.extend(pred)
        vids.extend(batch["video_id"])
        num_frames.extend((batch["num_segments"] * args.window_size).tolist())

    eval_results, plot_data = evaluate(args, val_loader, vids, predictions, num_frames, threshold, plot)

    if plot:
        save_path = f"{Path(args.ckpt_file).parents[1]}/plot"
        os.makedirs(save_path, exist_ok=True)
        print(f"Drawing results to {save_path}")
        for vid, data in tqdm(plot_data.items(), total=len(plot_data), dynamic_ncols=True):
            plot_result(save_path, vid, data["gt"], data["pred_matrix"])

    return eval_results

def evaluate(args, val_loader, vids, predictions, num_frames, threshold=0.5, plot=False):
    GT_FRAMES = val_loader.dataset.gt_frames
    GT_VIDEO_TYPES = val_loader.dataset.gt_videos
    crop_num = args.test_batch_size
    temp_pred = []
    plot_data = {}
    y_pred, y_true, y_pred_normal, cls_pred, cls_true = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    for i, (vid, num_frame, pred) in enumerate(zip(vids, num_frames, predictions)):
        vid = vid[:-3]
        video_type = GT_VIDEO_TYPES[vid][0]
        frame_gt = GT_FRAMES[vid]

        # 确保 frame_gt 和 pred 对齐
        if len(frame_gt) < num_frame:
            frame_gt = np.concatenate((frame_gt, np.full(num_frame - len(frame_gt), frame_gt[-1])))
        else:
            frame_gt = frame_gt[:num_frame]

        if len(pred) != num_frame:
            pred = pred[:num_frame] if len(pred) > num_frame else np.concatenate(
                (pred, np.full(num_frame - len(pred), pred[-1]))
            )

        temp_pred.append(pred)

        # 按批次处理
        if (i + 1) % crop_num == 0:
            ncrop_pred = np.mean(temp_pred, axis=0)
            y_pred = np.concatenate((y_pred, ncrop_pred))
            y_true = np.concatenate((y_true, frame_gt))

            if video_type == 0:
                y_pred_normal = np.concatenate((y_pred_normal, ncrop_pred))

            k = int(0.1 * num_frame)
            ncrop_cls_pred = np.partition(ncrop_pred, -k)[-k:].mean()
            cls_pred = np.concatenate((cls_pred, [ncrop_cls_pred]))
            cls_true = np.concatenate((cls_true, [video_type]))
            temp_pred.clear()

            if plot:
                plot_data[vid] = {"gt": frame_gt, "pred_matrix": np.array(ncrop_pred)}

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap_score = auc(recall, precision)
    far = cal_false_alarm(y_pred_normal, np.zeros(len(y_pred_normal)), threshold)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        frame_level_report = classification_report(y_true, np.where(y_pred >= threshold, 1, 0), zero_division=0)
        video_level_report = classification_report(cls_true, cls_pred.round(), zero_division=0)

    eval_results = {
        "AUC@PR": round(ap_score * 100, 4),
        "AUC@ROC": round(auc_score * 100, 4),
        "ACC": round(accuracy_score(cls_true, cls_pred.round()) * 100, 4),
        "FAR": round(far * 100, 4),
        "frame_level_report": frame_level_report,
        "video_level_report": video_level_report,
    }
    return eval_results, plot_data

if __name__ == "__main__":
    import importlib
    import sys

    for arg in sys.argv:
        if "ucfcrime" in str(arg):
            config_module = importlib.import_module("config.ucfcrime_cfg")
        elif "xdviolence" in str(arg):
            config_module = importlib.import_module("config.xdviolence_cfg")

    parser = config_module.parse_args()
    args_temp = parser.parse_args()
    assert os.path.isfile(args_temp.ckpt_file), f"{args_temp.ckpt_file} does not exist. Please specify the ckpt path!"

    args = torch.load(args_temp.ckpt_file, map_location=torch.device("cpu"))["args"]
    args.ckpt_file = args_temp.ckpt_file
    args.zip_feats = args_temp.zip_feats
    args.pseudo_label = "data/UCF_frame_pseudo_label.npy"

    setup_device(args)
    setup_seed(args)
    setup_dataset(args)

    eval(args, plot=False)