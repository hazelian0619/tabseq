import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tabseq.data.datasets import load_california_housing_split
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.metrics.holographic import ExtendedHolographicMetric
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.utils.seed import set_seed

def _resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        ckpt_path = os.path.join(path, 'checkpoint.pt')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f'checkpoint.pt not found in dir: {path}')
        return ckpt_path
    return path

def _infer_model_class(state_dict: dict):
    if any((key.startswith('decoder.') for key in state_dict.keys())):
        return TransformerTabSeqModel
    if any((key.startswith('tabular_encoder.') for key in state_dict.keys())):
        return TransformerTabSeqModel
    if any((key.startswith('encoder.') for key in state_dict.keys())):
        return TransformerTabSeqModel

def _greedy_step_probs(model: torch.nn.Module, x_num: torch.Tensor, depth: int, n_bins: int, temperature: float, sos_token: int=2) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError('temperature must be > 0')
    B = x_num.shape[0]
    device = x_num.device
    dec_input = torch.zeros((B, depth), dtype=torch.long, device=device)
    dec_input[:, 0] = sos_token
    step_probs_out = torch.empty((B, depth, n_bins), dtype=torch.float32, device=device)
    start = [0 for _ in range(B)]
    end = [n_bins for _ in range(B)]
    for t in range(depth):
        logits = model({'x_num': x_num, 'dec_input': dec_input})
        probs_t = torch.sigmoid(logits[:, t, :] / temperature)
        mask = torch.zeros_like(probs_t)
        for b in range(B):
            mask[b, start[b]:end[b]] = 1.0
        probs_t = probs_t * mask
        step_probs_out[:, t, :] = probs_t
        if t < depth - 1:
            bits = torch.empty((B,), dtype=torch.long, device=device)
            for b in range(B):
                s = start[b]
                e = end[b]
                mid = (s + e) // 2
                left = probs_t[b, s:mid].mean().item()
                right = probs_t[b, mid:e].mean().item()
                bit = 1 if right > left else 0
                bits[b] = bit
                if bit == 0:
                    end[b] = mid
                else:
                    start[b] = mid
            dec_input[:, t + 1] = bits
    return step_probs_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='例如：outputs/run_xxx/ 或 outputs/run_xxx/checkpoint.pt')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--random-state', type=int, default=0, help='要和 train.py 一致（默认 0）')
    ap.add_argument('--confidence', type=float, default=0.9, help='区间置信度，例如 0.90 表示 90%% 预测区间')
    ap.add_argument('--temperature', type=float, default=1.0, help='温度T：sigmoid(logits/T)，T>1 更保守')
    ap.add_argument('--mode', type=str, default='teacher_forcing', choices=['teacher_forcing', 'greedy', 'both'], help='teacher_forcing=用真实dec_input；greedy=真实推理只给SOS；both=两者都跑')
    args = ap.parse_args()
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    depth = int(cfg['depth'])
    n_bins = int(cfg['n_bins'])
    n_num_features = int(cfg['n_num_features'])
    v_min = float(cfg['v_min'])
    v_max = float(cfg['v_max'])
    set_seed(args.random_state)
    split = load_california_housing_split(random_state=args.random_state)
    X_val, y_val = (split.X_val, split.y_val)
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)
    ds = TabSeqDataset(X_num=X_val, X_cat=np.zeros((len(y_val), 0), dtype=np.int64), y=y_val, encoder=enc, is_train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model_cls = _infer_model_class(ckpt['model_state_dict'])
    if model_cls is TransformerTabSeqModel:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins, cat_cardinalities=cfg.get('cat_cardinalities'))
    else:
        model = model_cls(n_num_features=n_num_features, depth=depth, n_bins=n_bins)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    metric_calc = ExtendedHolographicMetric(enc)

    def run_mode(mode: str):
        step_probs_all = []
        y_true_all = []
        with torch.no_grad():
            for batch in dl:
                if mode == 'teacher_forcing':
                    logits = model(batch)
                    step_probs = torch.sigmoid(logits / float(args.temperature))
                elif mode == 'greedy':
                    step_probs = _greedy_step_probs(model=model, x_num=batch['x_num'], depth=depth, n_bins=n_bins, temperature=float(args.temperature), sos_token=2)
                else:
                    raise ValueError(f'unknown mode: {mode}')
                step_probs_all.append(step_probs.cpu())
                y_true_all.append(batch['y_raw'].cpu())
        metrics = metric_calc.compute_metrics(model_probs=torch.cat(step_probs_all, dim=0), y_true=torch.cat(y_true_all, dim=0), confidence=float(args.confidence), return_extras=True)
        metrics['model'] = cfg.get('model', 'tabseq')
        metrics['mode'] = mode
        metrics['temperature'] = float(args.temperature)
        return metrics
    out_dir = os.path.dirname(ckpt_path)
    if args.mode in ('teacher_forcing', 'both'):
        m_tf = run_mode('teacher_forcing')
        print('teacher_forcing:', m_tf)
        with open(os.path.join(out_dir, 'metrics_val_teacher_forcing.json'), 'w', encoding='utf-8') as f:
            json.dump(m_tf, f, ensure_ascii=False, indent=2)
    if args.mode in ('greedy', 'both'):
        m_g = run_mode('greedy')
        print('greedy:', m_g)
        with open(os.path.join(out_dir, 'metrics_val_greedy.json'), 'w', encoding='utf-8') as f:
            json.dump(m_g, f, ensure_ascii=False, indent=2)
    if args.mode == 'both':
        both = {'teacher_forcing': m_tf, 'greedy': m_g}
        out_path = os.path.join(out_dir, 'metrics_val.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(both, f, ensure_ascii=False, indent=2)
        print('saved:', out_path)
    else:
        out_path = os.path.join(out_dir, f'metrics_val_{args.mode}.json')
        print('saved:', out_path)
if __name__ == '__main__':
    main()
