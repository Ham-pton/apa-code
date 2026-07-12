import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.model_clean import HuPERAPA


UTT_ASPECTS = ("accuracy", "completeness", "fluency", "prosodic", "total")
WORD_ASPECTS = ("accuracy", "stress", "total")

GOP_MEAN = 3.203
GOP_STD = 4.045
ENERGY_MEAN = 0.1697
ENERGY_STD = 0.4824


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/seq_data_librispeech"))
    parser.add_argument("--exp-dir", type=Path, default=Path("exp/huper_apa"))

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--phone-loss-weight", type=float, default=1.0)
    parser.add_argument("--word-loss-weight", type=float, default=1.0)
    parser.add_argument("--utterance-loss-weight", type=float, default=1.0)

    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--gop-dim", type=int, default=84)
    parser.add_argument("--energy-dim", type=int, default=7)
    parser.add_argument("--huper-dim", type=int, default=5)
    parser.add_argument("--ssl-dim", type=int, default=1024)
    parser.add_argument("--duration-name", default="dur_feat")
    parser.add_argument("--huper-name", default="huper_feat")
    parser.add_argument("--ssl-name", default="w2v_300m_feat_v2")

    parser.add_argument("--no-duration", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--no-huper", action="store_true")
    parser.add_argument("--no-ssl", action="store_true")
    parser.add_argument("--unified-projection", action="store_true")
    parser.add_argument("--no-phone-relations", action="store_true")

    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--sequence-d-conv", type=int, default=4)
    parser.add_argument("--sequence-expand", type=int, default=2)
    parser.add_argument("--same-word-bias", type=float, default=1.0)
    parser.add_argument("--relation-scale", type=float, default=0.1)
    return parser


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def normalize_feature(feature, mask, mean, std):
    feature = torch.nan_to_num(feature)
    return ((feature - mean) / std) * mask


class PronunciationDataset(Dataset):
    def __init__(self, split, args):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        prefix = "tr" if split == "train" else "te"
        data_dir = args.data_dir.resolve()

        phone_label = self._load(data_dir / f"{prefix}_label_phn.npy")
        utterance_label = self._load(data_dir / f"{prefix}_label_utt.npy")
        word_label = self._load(data_dir / f"{prefix}_label_word.npy")
        gop = self._load(data_dir / f"{prefix}_feat.npy")

        if phone_label.ndim != 3 or phone_label.shape[-1] < 2:
            raise ValueError("phone labels must have shape [N, T, 2]")
        if gop.ndim != 3 or gop.shape[-1] != args.gop_dim:
            raise ValueError(f"GOP feature dimension must be {args.gop_dim}")
        if gop.shape[:2] != phone_label.shape[:2]:
            raise ValueError("GOP features and phone labels are not aligned")

        self.phone_label = torch.from_numpy(phone_label).float()
        self.utterance_label = torch.from_numpy(utterance_label).float() / 5.0
        self.word_label = torch.from_numpy(word_label).float()
        self.word_label[:, :, :3] /= 5.0

        valid_mask = (self.phone_label[:, :, :1] >= 0).float()
        feature_parts = [
            normalize_feature(torch.from_numpy(gop).float(), valid_mask, GOP_MEAN, GOP_STD)
        ]

        if not args.no_energy:
            energy = self._load_phone_feature(
                data_dir / f"{prefix}_energy_feat.npy",
                phone_label.shape[:2],
                args.energy_dim,
            )
            energy = normalize_feature(energy, valid_mask, ENERGY_MEAN, ENERGY_STD)
            feature_parts.append(energy)

        if not args.no_huper:
            huper = self._load_phone_feature(
                data_dir / f"{prefix}_{args.huper_name}.npy",
                phone_label.shape[:2],
                args.huper_dim,
            )
            feature_parts.append(torch.nan_to_num(huper) * valid_mask)

        if not args.no_ssl:
            ssl = self._load_phone_feature(
                data_dir / f"{prefix}_{args.ssl_name}.npy",
                phone_label.shape[:2],
                args.ssl_dim,
            )
            feature_parts.append(torch.nan_to_num(ssl) * valid_mask)

        self.features = torch.cat(feature_parts, dim=-1)

        if args.no_duration:
            self.duration = torch.zeros(*phone_label.shape[:2], 1)
        else:
            duration = self._load(data_dir / f"{prefix}_{args.duration_name}.npy")
            if duration.ndim == 2:
                duration = duration[..., None]
            if duration.ndim != 3 or duration.shape[:2] != phone_label.shape[:2]:
                raise ValueError("duration features must have shape [N, T] or [N, T, D]")
            self.duration = torch.nan_to_num(torch.from_numpy(duration).float()) * valid_mask

        self.input_dim = self.features.shape[-1]
        self.duration_dim = self.duration.shape[-1]

        print(
            f"{split}: features={tuple(self.features.shape)}, "
            f"duration={tuple(self.duration.shape)}"
        )

    @staticmethod
    def _load(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.load(path, allow_pickle=False)

    @classmethod
    def _load_phone_feature(cls, path, expected_shape, expected_dim):
        array = cls._load(path)
        if array.ndim != 3:
            raise ValueError(f"{path.name} must have shape [N, T, D]")
        if array.shape[:2] != tuple(expected_shape):
            raise ValueError(f"{path.name} is not aligned with the phone sequence")
        if array.shape[-1] != expected_dim:
            raise ValueError(
                f"{path.name} has dimension {array.shape[-1]}, expected {expected_dim}"
            )
        return torch.from_numpy(array).float()

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        phone = self.phone_label[index]
        word = self.word_label[index]
        return (
            self.features[index],
            phone[:, 1],
            phone[:, 0].long(),
            self.utterance_label[index],
            word,
            word[:, -1].long(),
            self.duration[index],
        )


def masked_mse(prediction, target, mask):
    mask = mask.bool()
    if not mask.any():
        return prediction.new_tensor(0.0)
    return ((prediction - target) ** 2)[mask].mean()


def model_forward(model, batch, device):
    features, phone_target, phone_id, utt_target, word_target, word_id, duration = batch
    features = features.to(device, non_blocking=True)
    phone_id = phone_id.to(device, non_blocking=True)
    word_id = word_id.to(device, non_blocking=True)
    duration = duration.to(device, non_blocking=True)

    outputs = model(features, phone_id, word_id, duration)
    targets = (
        phone_target.to(device, non_blocking=True),
        utt_target.to(device, non_blocking=True),
        word_target.to(device, non_blocking=True),
    )
    return outputs, targets


def split_outputs(outputs):
    utterance = torch.cat(outputs[:5], dim=1)
    phone = outputs[5].squeeze(-1)
    word = torch.cat(outputs[6:9], dim=-1)
    return phone, utterance, word


def compute_loss(outputs, targets, args):
    phone_pred, utt_pred, word_pred = split_outputs(outputs)
    phone_target, utt_target, word_target = targets

    phone_loss = masked_mse(phone_pred, phone_target, phone_target >= 0)
    word_scores = word_target[:, :, :3]
    word_loss = masked_mse(word_pred, word_scores, word_scores >= 0)
    utt_loss = nn.functional.mse_loss(utt_pred, utt_target)

    total = (
        args.phone_loss_weight * phone_loss
        + args.word_loss_weight * word_loss
        + args.utterance_loss_weight * utt_loss
    )
    return total, phone_loss, word_loss, utt_loss


def pearson_corr(prediction, target):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.size < 2 or prediction.std() == 0 or target.std() == 0:
        return float("nan")
    return float(np.corrcoef(prediction, target)[0, 1])


def phone_metrics(prediction, target):
    prediction = prediction.squeeze(-1).numpy()
    target = target.numpy()
    mask = target >= 0
    prediction = prediction[mask]
    target = target[mask]
    return float(np.mean((prediction - target) ** 2)), pearson_corr(prediction, target)


def utterance_metrics(prediction, target):
    prediction = prediction.numpy()
    target = target.numpy()
    mse = [float(np.mean((prediction[:, i] - target[:, i]) ** 2)) for i in range(5)]
    pcc = [pearson_corr(prediction[:, i], target[:, i]) for i in range(5)]
    return mse, pcc


def collect_word_scores(prediction, target):
    prediction = prediction.numpy()
    target = target.numpy()
    word_ids = target[:, :, -1].astype(np.int64)
    scores = target[:, :, :3]

    word_prediction = []
    word_target = []
    for sample_pred, sample_score, sample_ids in zip(prediction, scores, word_ids):
        start = 0
        while start < len(sample_ids) and sample_ids[start] >= 0:
            end = start + 1
            while end < len(sample_ids) and sample_ids[end] == sample_ids[start]:
                end += 1
            word_prediction.append(sample_pred[start:end].mean(axis=0))
            word_target.append(sample_score[start:end].mean(axis=0))
            start = end

    return np.asarray(word_prediction), np.asarray(word_target).round(2)


def word_metrics(prediction, target):
    word_prediction, word_target = collect_word_scores(prediction, target)
    mse = [
        float(np.mean((word_prediction[:, i] - word_target[:, i]) ** 2))
        for i in range(3)
    ]
    pcc = [pearson_corr(word_prediction[:, i], word_target[:, i]) for i in range(3)]
    return mse, pcc, word_prediction, word_target


def evaluate(model, loader, device):
    model.eval()
    phone_pred, phone_target = [], []
    utt_pred, utt_target = [], []
    word_pred, word_target = [], []

    with torch.no_grad():
        for batch in loader:
            outputs, targets = model_forward(model, batch, device)
            phone, utterance, word = split_outputs(outputs)

            phone_pred.append(phone.cpu().unsqueeze(-1))
            phone_target.append(targets[0].cpu())
            utt_pred.append(utterance.cpu())
            utt_target.append(targets[1].cpu())
            word_pred.append(word.cpu())
            word_target.append(targets[2].cpu())

    arrays = {
        "phone_pred": torch.cat(phone_pred),
        "phone_target": torch.cat(phone_target),
        "utt_pred": torch.cat(utt_pred),
        "utt_target": torch.cat(utt_target),
        "word_pred": torch.cat(word_pred),
        "word_target": torch.cat(word_target),
    }

    phn_mse, phn_pcc = phone_metrics(arrays["phone_pred"], arrays["phone_target"])
    utt_mse, utt_pcc = utterance_metrics(arrays["utt_pred"], arrays["utt_target"])
    word_mse, word_pcc, valid_word_pred, valid_word_target = word_metrics(
        arrays["word_pred"], arrays["word_target"]
    )

    arrays["valid_word_pred"] = valid_word_pred
    arrays["valid_word_target"] = valid_word_target
    metrics = {
        "phone_mse": phn_mse,
        "phone_pcc": phn_pcc,
        "utterance_mse": utt_mse,
        "utterance_pcc": utt_pcc,
        "word_mse": word_mse,
        "word_pcc": word_pcc,
    }
    return metrics, arrays


def result_header():
    header = [
        "epoch",
        "learning_rate",
        "phone_train_mse",
        "phone_train_pcc",
        "phone_test_mse",
        "phone_test_pcc",
    ]
    for split in ("train", "test"):
        for metric in ("mse", "pcc"):
            header.extend(f"utterance_{split}_{metric}_{name}" for name in UTT_ASPECTS)
    for split in ("train", "test"):
        header.extend(f"word_{split}_pcc_{name}" for name in WORD_ASPECTS)
    return header


def result_row(epoch, lr, train_metrics, test_metrics):
    row = [
        epoch,
        lr,
        train_metrics["phone_mse"],
        train_metrics["phone_pcc"],
        test_metrics["phone_mse"],
        test_metrics["phone_pcc"],
    ]
    for metrics in (train_metrics, test_metrics):
        row.extend(metrics["utterance_mse"])
        row.extend(metrics["utterance_pcc"])
    row.extend(train_metrics["word_pcc"])
    row.extend(test_metrics["word_pcc"])
    return row


def save_results(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(result_header())
        writer.writerows(rows)


def save_predictions(exp_dir, arrays):
    pred_dir = exp_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "phone_pred.npy", arrays["phone_pred"].numpy())
    np.save(pred_dir / "phone_target.npy", arrays["phone_target"].numpy())
    np.save(pred_dir / "word_pred.npy", arrays["valid_word_pred"])
    np.save(pred_dir / "word_target.npy", arrays["valid_word_target"])
    np.save(pred_dir / "utterance_pred.npy", arrays["utt_pred"].numpy())
    np.save(pred_dir / "utterance_target.npy", arrays["utt_target"].numpy())


def train(model, train_loader, train_eval_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.exp_dir / "models"
    model_dir.mkdir(exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-7,
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_phone_mse = float("inf")
    global_step = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        loss_sum = np.zeros(4, dtype=np.float64)
        batch_count = 0

        for batch in train_loader:
            if args.warmup_steps > 0 and global_step < args.warmup_steps:
                warmup_lr = args.lr * (global_step + 1) / args.warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = warmup_lr

            outputs, targets = model_forward(model, batch, device)
            losses = compute_loss(outputs, targets, args)

            optimizer.zero_grad(set_to_none=True)
            losses[0].backward()
            optimizer.step()

            loss_sum += np.array([loss.detach().item() for loss in losses])
            batch_count += 1
            global_step += 1

        train_metrics, _ = evaluate(model, train_eval_loader, device)
        test_metrics, test_arrays = evaluate(model, test_loader, device)

        monitor = (
            test_metrics["phone_mse"]
            + np.mean(test_metrics["word_mse"])
            + np.mean(test_metrics["utterance_mse"])
        )
        if global_step >= args.warmup_steps:
            scheduler.step(monitor)

        lr = optimizer.param_groups[0]["lr"]
        rows.append(result_row(epoch, lr, train_metrics, test_metrics))
        save_results(args.exp_dir / "result.csv", rows)

        if test_metrics["phone_mse"] < best_phone_mse:
            best_phone_mse = test_metrics["phone_mse"]
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, model_dir / "best_model.pth")
            save_predictions(args.exp_dir, test_arrays)

        avg_loss = loss_sum / max(batch_count, 1)
        elapsed = int(time.time() - start_time)
        print(
            f"epoch {epoch:03d}/{args.epochs} | {elapsed:4d}s | "
            f"loss {avg_loss[0]:.4f} "
            f"(phn {avg_loss[1]:.4f}, word {avg_loss[2]:.4f}, utt {avg_loss[3]:.4f})"
        )
        print(
            f"  phone: MSE {test_metrics['phone_mse']:.3f}, "
            f"PCC {test_metrics['phone_pcc']:.3f}"
        )
        print(
            "  word PCC: "
            + ", ".join(
                f"{name} {score:.3f}"
                for name, score in zip(WORD_ASPECTS, test_metrics["word_pcc"])
            )
        )
        print(
            "  utterance PCC: "
            + ", ".join(
                f"{name} {score:.3f}"
                for name, score in zip(UTT_ASPECTS, test_metrics["utterance_pcc"])
            )
        )


def make_loader(dataset, batch_size, shuffle, args, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if shuffle else None,
        generator=generator,
    )


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    train_set = PronunciationDataset("train", args)
    test_set = PronunciationDataset("test", args)
    if train_set.input_dim != test_set.input_dim:
        raise ValueError("train and test feature dimensions do not match")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(train_set, args.batch_size, True, args, generator)
    train_eval_loader = make_loader(train_set, args.eval_batch_size, False, args)
    test_loader = make_loader(test_set, args.eval_batch_size, False, args)

    model = HuPERAPA(
        embed_dim=args.embed_dim,
        depth=args.depth,
        input_dim=train_set.input_dim,
        num_heads=args.num_heads,
        dur_dim=train_set.duration_dim,
        use_dur=not args.no_duration,
        sequence_layers=args.sequence_layers,
        sequence_d_conv=args.sequence_d_conv,
        sequence_expand=args.sequence_expand,
        sequence_gate=True,
        use_word_sequence=True,
        use_adaptive_phone_graph=not args.no_phone_relations,
        phone_graph_same_word_bias=args.same_word_bias,
        phone_graph_scale=args.relation_scale,
        gop_dim=args.gop_dim,
        energy_dim=args.energy_dim,
        huper_dim=args.huper_dim,
        ssl_dim=args.ssl_dim,
        use_energy=not args.no_energy,
        use_huper=not args.no_huper,
        use_ssl=not args.no_ssl,
        use_evidence_fusion=not args.unified_projection,
    )

    train(model, train_loader, train_eval_loader, test_loader, args)


if __name__ == "__main__":
    main()
