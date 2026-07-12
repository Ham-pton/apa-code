import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WavLMForCTC


PHONE_MAP = {
    "AO": "AA",
    "AX": "AH",
    "AXR": "ER",
    "IX": "IH",
    "EL": "L",
    "EM": "M",
    "EN": "N",
}


def normalize_phone(phone):
    phone = str(phone).strip().upper().split("_", 1)[0]
    return re.sub(r"\d+$", "", phone)


def huper_phone(phone):
    phone = normalize_phone(phone)
    return PHONE_MAP.get(phone, phone)


def utterance_id(item):
    item = str(item).strip()
    return item.rsplit(".", 1)[0] if "." in item else item


def numeric_id(item):
    digits = re.sub(r"\D", "", str(item))
    return str(int(digits)) if digits else ""


def read_order(path):
    order = []
    seen = set()

    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue

            item = row[0].strip()
            if not item or not re.search(r"\d", item):
                continue

            utt_id = utterance_id(item)
            if utt_id not in seen:
                seen.add(utt_id)
                order.append(utt_id)

    return order


def read_phone_sequences(path):
    grouped = defaultdict(list)

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 or not re.search(r"\d", parts[0]):
                continue

            key = parts[0]
            utt_id = utterance_id(key)

            try:
                index = int(key.rsplit(".", 1)[1])
            except (IndexError, ValueError):
                index = len(grouped[utt_id])

            grouped[utt_id].append(
                (index, [normalize_phone(phone) for phone in parts[1:]])
            )

    sequences = {}
    for utt_id, items in grouped.items():
        phones = []
        for _, word_phones in sorted(items, key=lambda x: x[0]):
            phones.extend(word_phones)
        sequences[utt_id] = phones

    return sequences


def index_audio(audio_root):
    exact = {}
    numeric = defaultdict(list)

    wav_files = list(Path(audio_root).rglob("*.wav"))
    wav_files += list(Path(audio_root).rglob("*.WAV"))

    for path in wav_files:
        exact.setdefault(path.stem, path)
        key = numeric_id(path.stem)
        if key:
            numeric[key].append(path)

    print(f"Found {len(wav_files)} wav files")
    return exact, numeric


def find_audio(utt_id, exact, numeric):
    if utt_id in exact:
        return exact[utt_id]

    matches = numeric.get(numeric_id(utt_id), [])
    if len(matches) > 1:
        print(f"Warning: multiple wav files matched {utt_id}; using {matches[0]}")
    return matches[0] if matches else None


def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )

    return waveform.squeeze(0)


def build_label_map(model, processor):
    label_to_id = {}

    id2label = getattr(model.config, "id2label", {}) or {}
    for index, label in id2label.items():
        label_to_id[normalize_phone(label)] = int(index)

    for label, index in processor.tokenizer.get_vocab().items():
        label_to_id[normalize_phone(label)] = int(index)

    return label_to_id


def get_blank_id(model, processor):
    if processor.tokenizer.pad_token_id is not None:
        return processor.tokenizer.pad_token_id
    if getattr(model.config, "pad_token_id", None) is not None:
        return model.config.pad_token_id
    return 0


def ctc_viterbi_align(log_probs, target_ids, blank_id):
    num_frames = log_probs.size(0)
    if not target_ids:
        return [-1] * num_frames

    states = [blank_id]
    for phone_id in target_ids:
        states.extend((phone_id, blank_id))

    states = torch.tensor(states, device=log_probs.device)
    num_states = states.numel()
    neg_inf = torch.finfo(log_probs.dtype).min

    score = torch.full(
        (num_states,), neg_inf, device=log_probs.device, dtype=log_probs.dtype
    )
    score[0] = log_probs[0, states[0]]
    if num_states > 1:
        score[1] = log_probs[0, states[1]]

    backptr = torch.zeros(
        (num_frames, num_states), device=log_probs.device, dtype=torch.long
    )

    skip_allowed = torch.zeros(
        num_states, device=log_probs.device, dtype=torch.bool
    )
    if num_states > 2:
        skip_allowed[2:] = (
            (states[2:] != blank_id) & (states[2:] != states[:-2])
        )

    for frame in range(1, num_frames):
        stay = score
        step = torch.cat((score.new_full((1,), neg_inf), score[:-1]))
        skip = torch.cat((score.new_full((2,), neg_inf), score[:-2]))
        skip = torch.where(skip_allowed, skip, skip.new_full(skip.shape, neg_inf))

        candidates = torch.stack((stay, step, skip))
        best_score, move = candidates.max(dim=0)

        state_index = torch.arange(num_states, device=log_probs.device)
        backptr[frame] = state_index - move
        score = best_score + log_probs[frame, states]

    last_state = 0
    if num_states > 1:
        last_state = (
            num_states - 1
            if score[-1] > score[-2]
            else num_states - 2
        )

    path = []
    state = int(last_state)
    for frame in range(num_frames - 1, -1, -1):
        path.append(state)
        state = int(backptr[frame, state])

    path.reverse()
    return [state // 2 if state % 2 else -1 for state in path]


@torch.inference_mode()
def extract_utterance(
    wav_path,
    phones,
    model,
    processor,
    label_to_id,
    blank_id,
    device,
    max_phones=50,
    pad_value=-1.0,
    fp16=False,
):
    features = np.full((max_phones, 5), pad_value, dtype=np.float32)
    waveform = load_audio(wav_path)

    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    if fp16 and device.type == "cuda":
        inputs["input_values"] = inputs["input_values"].half()

    logits = model(**inputs).logits[0].float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    target_ids = []
    target_positions = []

    for position, phone in enumerate(phones[:max_phones]):
        mapped_phone = huper_phone(phone)
        if mapped_phone in label_to_id:
            target_ids.append(label_to_id[mapped_phone])
            target_positions.append(position)
        else:
            print(
                f"Warning: phone {phone} (mapped to {mapped_phone}) "
                f"is not in the HuPER label set"
            )

    if not target_ids:
        return features

    alignment = ctc_viterbi_align(log_probs, target_ids, blank_id)
    frames_by_phone = defaultdict(list)

    for frame, target_index in enumerate(alignment):
        if target_index >= 0:
            frames_by_phone[target_index].append(frame)

    top2_prob, top2_id = probs.topk(2, dim=-1)
    top1_prob = top2_prob[:, 0]
    margin = top2_prob[:, 0] - top2_prob[:, 1]
    top1_id = top2_id[:, 0]

    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / np.log(probs.size(-1))

    for target_index, output_position in enumerate(target_positions):
        frames = frames_by_phone.get(target_index)
        if not frames:
            continue

        start, end = frames[0], frames[-1] + 1
        canonical_id = target_ids[target_index]
        current_probs = probs[start:end]

        features[output_position] = [
            current_probs[:, canonical_id].mean().item(),
            top1_prob[start:end].mean().item(),
            margin[start:end].mean().item(),
            entropy[start:end].mean().item(),
            (top1_id[start:end] == canonical_id).float().mean().item(),
        ]

    return features


def write_report(path, split, array, missing_audio, missing_phones):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"split: {split}\n")
        f.write(f"feature_shape: {array.shape}\n")
        f.write(f"missing_audio: {len(missing_audio)}\n")
        f.write(f"missing_phone_sequence: {len(missing_phones)}\n")

        if missing_audio:
            f.write("\n[missing_audio]\n")
            f.write("\n".join(missing_audio))

        if missing_phones:
            f.write("\n\n[missing_phone_sequence]\n")
            f.write("\n".join(missing_phones))


def extract_split(
    split,
    order,
    phone_sequences,
    exact_audio,
    numeric_audio,
    model,
    processor,
    label_to_id,
    blank_id,
    device,
    output_dir,
    max_phones,
    split_size,
    pad_value,
    fp16,
):
    if len(order) != split_size:
        raise ValueError(
            f"{split} contains {len(order)} utterances; expected {split_size}"
        )

    features = []
    missing_audio = []
    missing_phones = []

    for utt_id in tqdm(order, desc=f"Extract {split}"):
        wav_path = find_audio(utt_id, exact_audio, numeric_audio)
        phones = phone_sequences.get(utt_id)

        if wav_path is None:
            missing_audio.append(utt_id)
            feat = np.full((max_phones, 5), pad_value, dtype=np.float32)
        elif phones is None:
            missing_phones.append(utt_id)
            feat = np.full((max_phones, 5), pad_value, dtype=np.float32)
        else:
            feat = extract_utterance(
                wav_path,
                phones,
                model,
                processor,
                label_to_id,
                blank_id,
                device,
                max_phones,
                pad_value,
                fp16,
            )

        features.append(feat)

    array = np.stack(features)
    expected_shape = (split_size, max_phones, 5)
    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected {split} feature shape: {array.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split}_huper_feat.npy", array)

    with open(output_dir / f"{split}_uttids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(order))

    write_report(
        output_dir / f"{split}_huper_report.txt",
        split,
        array,
        missing_audio,
        missing_phones,
    )

    print(f"Saved {split}: {array.shape}")
    return array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract HuPER phoneme-posterior perceptual features."
    )
    parser.add_argument("--audio-root", required=True)
    parser.add_argument(
        "--train-order",
        default="data/raw_kaldi_gop/librispeech/tr_keys_phn.csv",
    )
    parser.add_argument(
        "--test-order",
        default="data/raw_kaldi_gop/librispeech/te_keys_phn.csv",
    )
    parser.add_argument("--phone-text", required=True)
    parser.add_argument(
        "--output-dir", default="data/seq_data_librispeech"
    )
    parser.add_argument(
        "--repo-id", default="huper29/huper_recognizer"
    )
    parser.add_argument("--max-phones", type=int, default=50)
    parser.add_argument("--split-size", type=int, default=2500)
    parser.add_argument("--pad-value", type=float, default=-1.0)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_order = read_order(args.train_order)
    test_order = read_order(args.test_order)
    phone_sequences = read_phone_sequences(args.phone_text)
    exact_audio, numeric_audio = index_audio(args.audio_root)

    print(f"Device: {device}")
    print(f"Train utterances: {len(train_order)}")
    print(f"Test utterances: {len(test_order)}")
    print(f"Phone sequences: {len(phone_sequences)}")

    processor = Wav2Vec2Processor.from_pretrained(args.repo_id)
    model = WavLMForCTC.from_pretrained(args.repo_id)

    if args.fp16 and device.type == "cuda":
        model = model.half()

    model = model.to(device).eval()
    label_to_id = build_label_map(model, processor)
    blank_id = get_blank_id(model, processor)
    output_dir = Path(args.output_dir)

    common = dict(
        phone_sequences=phone_sequences,
        exact_audio=exact_audio,
        numeric_audio=numeric_audio,
        model=model,
        processor=processor,
        label_to_id=label_to_id,
        blank_id=blank_id,
        device=device,
        output_dir=output_dir,
        max_phones=args.max_phones,
        split_size=args.split_size,
        pad_value=args.pad_value,
        fp16=args.fp16,
    )

    extract_split("tr", train_order, **common)
    extract_split("te", test_order, **common)


if __name__ == "__main__":
    main()
