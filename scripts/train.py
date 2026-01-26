import argparse
import json
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    import swanlab
except Exception:
    swanlab = None
from tabseq.data.datasets import load_california_housing_split
from tabseq.labels.trace_encoder import TraceLabelEncoder
from tabseq.data.tabseq_dataset import TabSeqDataset
from tabseq.models.transformer_model import TransformerTabSeqModel
from tabseq.utils.git import get_git_hash
from tabseq.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--depth', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--no-confidence-masking', action='store_true')
    ap.add_argument('--alpha-depth-mode', type=str, default=None)
    ap.add_argument('--alpha-min', type=float, default=None)
    ap.add_argument('--alpha-max', type=float, default=None)
    ap.add_argument('--out-dir', type=str, default=None)
    args = ap.parse_args()
    seed = int(args.seed) if args.seed is not None else int(os.environ.get('TABSEQ_SEED', '0'))
    depth = int(args.depth) if args.depth is not None else 6
    n_bins = 2 ** depth
    use_confidence_masking = not args.no_confidence_masking
    alpha_depth_mode = args.alpha_depth_mode or 'linear'
    alpha_min = float(args.alpha_min) if args.alpha_min is not None else 0.2
    alpha_max = float(args.alpha_max) if args.alpha_max is not None else 0.8
    if not 0.0 <= alpha_min <= alpha_max <= 1.0:
        raise ValueError('alpha_min/alpha_max must satisfy 0<=alpha_min<=alpha_max<=1')
    n_train = 256
    batch_size = int(args.batch_size) if args.batch_size is not None else 32
    epochs = int(args.epochs) if args.epochs is not None else 2
    lr = float(args.lr) if args.lr is not None else 0.001
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join('outputs', f'run_{run_id}')
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    run_cfg = {'seed': seed, 'depth': depth, 'n_bins': n_bins, 'n_train': n_train, 'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'use_confidence_masking': use_confidence_masking, 'alpha_depth_mode': alpha_depth_mode, 'alpha_min': alpha_min, 'alpha_max': alpha_max}
    if swanlab is not None:
        try:
            swanlab.init(project='tabseq-trace', experiment=run_id, config=run_cfg)
        except TypeError:
            swanlab.init(project='tabseq-trace', config=run_cfg)
    split = load_california_housing_split(random_state=seed)
    X_num = split.X_train
    y_train = split.y_train
    n_num_features = X_num.shape[1]
    X_cat = np.zeros((len(y_train), 0), dtype=np.int64)
    v_min, v_max = (float(y_train.min()), float(y_train.max()))
    enc = TraceLabelEncoder(v_min=v_min, v_max=v_max, depth=depth)
    ds = TabSeqDataset(X_num=X_num, X_cat=X_cat, y=y_train, encoder=enc)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = TransformerTabSeqModel(n_num_features=n_num_features, depth=depth, n_bins=n_bins)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    if depth <= 1:
        alpha_depth = torch.full((depth,), alpha_max)
    elif alpha_depth_mode == 'linear':
        steps = torch.linspace(0.0, 1.0, steps=depth)
        alpha_depth = alpha_min + (alpha_max - alpha_min) * steps
    else:
        raise ValueError(f'unknown alpha_depth_mode: {alpha_depth_mode}')
    alpha_depth_values = alpha_depth.tolist()
    alpha_depth = alpha_depth.to(next(model.parameters()).device)
    alpha_depth_view = alpha_depth.view(1, depth, 1)
    model.train()
    step = 0
    alpha_sum = 0.0
    alpha_count = 0
    alpha_instance_min = None
    alpha_instance_max = None
    for epoch in range(epochs):
        for batch in dl:
            if use_confidence_masking:
                logits, ctx_tokens = model(batch, return_context=True)
                alpha_instance = model.compute_alpha_instance(ctx_tokens)
            else:
                logits = model(batch)
                alpha_instance = None
            target = batch['y_mht']
            loss_raw = loss_fn(logits, target)
            if use_confidence_masking:
                alpha = alpha_depth_view * alpha_instance.view(-1, 1, 1)
                alpha = alpha.clamp(0.0, 1.0)
                weight = torch.where(target > 0.5, torch.ones_like(loss_raw), 1.0 - alpha)
                loss = (loss_raw * weight).mean()
                alpha_sum += float(alpha_instance.sum().item())
                alpha_count += int(alpha_instance.numel())
                batch_min = float(alpha_instance.min().item())
                batch_max = float(alpha_instance.max().item())
                alpha_instance_min = batch_min if alpha_instance_min is None else min(alpha_instance_min, batch_min)
                alpha_instance_max = batch_max if alpha_instance_max is None else max(alpha_instance_max, batch_max)
            else:
                loss = loss_raw.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0:
                print(f'epoch={epoch} step={step} loss={loss.item():.6f}')
                if swanlab is not None:
                    log_data = {'loss': float(loss.item()), 'epoch': int(epoch), 'step': int(step)}
                    if use_confidence_masking:
                        log_data['alpha_instance_mean'] = float(alpha_instance.mean().item())
                        log_data['alpha_instance_min'] = float(alpha_instance.min().item())
                        log_data['alpha_instance_max'] = float(alpha_instance.max().item())
                    swanlab.log(log_data)
            step += 1
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    cfg = {'v_min': v_min, 'v_max': v_max, 'depth': depth, 'n_bins': n_bins, 'n_num_features': n_num_features, 'seed': seed, 'cat_cardinalities': [], 'model': 'tabseq_daca', 'use_confidence_masking': use_confidence_masking, 'alpha_depth_mode': alpha_depth_mode, 'alpha_min': alpha_min, 'alpha_max': alpha_max, 'alpha_depth_values': alpha_depth_values}
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, ckpt_path)
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'git.txt'), 'w', encoding='utf-8') as f:
        f.write(get_git_hash() + '\n')
    metrics = {'final_loss': float(loss.item())}
    if use_confidence_masking and alpha_count > 0:
        metrics['alpha_instance_mean'] = float(alpha_sum / alpha_count)
        metrics['alpha_instance_min'] = float(alpha_instance_min)
        metrics['alpha_instance_max'] = float(alpha_instance_max)
    with open(os.path.join(out_dir, 'metrics_train.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print('saved:', ckpt_path)
if __name__ == '__main__':
    main()
