"""Run A-E sensitivity-tiered channel pruning with one shared implementation.

Training-only FP32 mean(|W*dL/dW|) scores are a heuristic, not an accuracy
guarantee. Experiment03's L1 masking tiers are reused as a proxy. Temporary
checkpoints are removed only after a selected checkpoint is verified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.metadata
import math
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.modules import C2f, C2PSA
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import init_seeds

from greedy_evaluation import evaluate, run_finetune, benchmark
from prune_greedy import (ROOT, Tee, now, sha, write_json, fingerprint,
                          shape_signature, stats, save_model, backward_check)

SCHEMES = {"A": (.125, 0., 0.), "B": (.20, 0., 0.), "C": (.25, 0., 0.),
           "D": (.25, .10, 0.), "E": (.25, .15, .05)}
TIERS = ("low", "medium", "high")


def load_tiers(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        name, width = row["layer"], int(row["out_channels"])
        if name.startswith(("model.0.", "model.10.", "model.23.")) or width < 64:
            continue
        if name == "model.13.m.0.cv2.conv":
            continue  # Known unsupported CSP dependency, recorded in the report.
        drop = float(row["mAP50_95_drop"])
        tier = "low" if drop <= .005 else "medium" if drop <= .020 else "high"
        result.append(dict(layer=name, initial_channels=width, drop=drop, tier=tier))
    return sorted(result, key=lambda r: (r["drop"], r["layer"]))


def collect_scores(base, data_path, tiers, device):
    """One pass over ALL training images; no optimizer or AMP accumulation."""
    init_seeds(42, deterministic=True)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 640, "rect": False})
    data = check_det_dataset(str(data_path), autodownload=False)
    dataset = build_yolo_dataset(cfg, data["train"], 8, data, mode="val", rect=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                       num_workers=0, collate_fn=dataset.collate_fn)
    model = YOLO(str(base)).model.float().to(device)
    before = fingerprint(model)
    model.args = get_cfg(overrides=dict(model.args))
    model.criterion = None
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    modules = dict(model.named_modules())
    scores = {r["layer"]: torch.zeros(r["initial_channels"], dtype=torch.float64) for r in tiers}
    images, losses, files = 0, [], []
    started = time.perf_counter()
    for index, batch in enumerate(loader, 1):
        files.extend(batch["im_file"])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float() / 255.
        size = batch["img"].shape[0]
        model.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss = loss.sum() / size
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite calibration loss")
        loss.backward()
        for name, total in scores.items():
            weight = modules[name].weight
            if weight.grad is None or not torch.isfinite(weight.grad).all():
                raise RuntimeError(f"missing/nonfinite gradient: {name}")
            value = (weight.detach() * weight.grad.detach()).abs().flatten(1).mean(1)
            total.add_(value.double().cpu(), alpha=size)
        images += size
        losses.append(float(loss.detach()))
        print(f"CALIBRATE {index}/{len(loader)} images={images} loss={losses[-1]:.5f}", flush=True)
    assert fingerprint(model) == before, "calibration changed checkpoint tensors or BN"
    assert images == len(dataset) == len(set(files)), "calibration coverage mismatch"
    values = {k: (v / images).tolist() for k, v in scores.items()}
    assert all(all(math.isfinite(x) for x in v) and max(v) > 0 for v in values.values())
    summary = dict(images=images, batches=len(loader), files=files, losses=losses,
                   seconds=time.perf_counter()-started, unchanged_checkpoint=True,
                   precision="FP32", scores=values)
    del model, loader, dataset, batch, loss, modules
    gc.collect()
    torch.cuda.empty_cache()
    return values, summary


def prune_action(model, name, indices, identities, example, initial):
    """Mutate a disposable copy only; keep original channel IDs across groups."""
    model.eval().float()
    for p in model.parameters():
        p.requires_grad_(True)
    modules = dict(model.named_modules())
    reverse = {m: n for n, m in modules.items()}
    split_outputs = {m.cv1.conv for m in model.modules() if isinstance(m, (C2f, C2PSA))}
    root = modules[name]
    before = shape_signature(model)
    graph = tp.DependencyGraph().build_dependency(model, example_inputs=example)
    group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=indices)
    if not graph.check_pruning_group(group):
        raise ValueError("invalid_dependency_group")
    output_deletions, operations = {}, []
    for dep, idxs in group:
        module = dep.target.module
        n = reverse.get(module, "")
        out = graph.is_out_channel_pruning_fn(dep.handler)
        if module in split_outputs and out:
            raise ValueError(f"protect_CSP_chunk_width:{n}")
        if n == "model.0" or n.startswith(("model.0.", "model.10.", "model.23.dfl")):
            raise ValueError(f"protect_stem_attention_DFL:{n}")
        if isinstance(module, torch.nn.Conv2d):
            if n.startswith("model.23.") and n.count(".") == 4 and out:
                raise ValueError(f"protect_final_detection_output:{n}")
            ids = sorted(set(int(i) for i in idxs))
            if out:
                output_deletions.setdefault(n, set()).update(ids)
                if module.out_channels - len(output_deletions[n]) < max(8, initial[n] // 2):
                    raise ValueError(f"minimum_half_channels:{n}")
            operations.append(dict(layer=n, direction="out" if out else "in", indices=ids))
    updated = copy.deepcopy(identities)
    for n, removed in output_deletions.items():
        updated[n] = [old for i, old in enumerate(identities[n]) if i not in removed]
    old_width = root.out_channels
    original_removed = [identities[name][i] for i in indices]
    group.prune()
    for n, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and len(updated[n]) != module.out_channels:
            raise ValueError(f"channel_identity_mismatch:{n}")
    after = shape_signature(model)
    changes = {k: dict(before=before[k], after=after[k]) for k in before if before[k] != after[k]}
    if not changes:
        raise ValueError("no_shape_change")
    for side in (320, 640):
        with torch.no_grad():
            pred = model(torch.zeros(1, 3, side, side, device=example.device))[0]
        if pred.shape[1] != 84 or not torch.isfinite(pred).all():
            raise ValueError("invalid_prediction")
    model.zero_grad(set_to_none=True)
    return updated, dict(before_channels=old_width, after_channels=root.out_channels,
                         indices=indices, original_indices=original_removed,
                         dependencies=operations, changed_tensors=changes)


def apply_scheme(label, base, tiers, scores, example, work):
    model = YOLO(str(base)).model.float().to(example.device).eval()
    initial = {n: m.out_channels for n, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}
    identities = {n: list(range(w)) for n, w in initial.items()}
    ratios = dict(zip(TIERS, SCHEMES[label]))
    records = []
    for item in tiers:
        name = item["layer"]
        ratio = ratios[item["tier"]]
        if ratio == 0:
            continue
        # Target is cumulative relative to the original width, including deletion
        # caused by prior residual/depthwise dependencies; never prune twice.
        target = int(math.floor(initial[name] * ratio + .5))
        remaining = identities[name]
        count = target - (initial[name] - len(remaining))
        row = {**item, "requested_ratio": ratio, "target_removed": target}
        if count <= 0:
            row.update(status="already_met_by_dependency")
            records.append(row)
            continue
        ranking = sorted(range(len(remaining)), key=lambda i: (scores[name][remaining[i]], remaining[i]))
        idxs = sorted(ranking[:count])
        trial = copy.deepcopy(model)
        try:
            updated, change = prune_action(trial, name, idxs, identities, example, initial)
            row.update(status="accepted", **change)
            model, identities = trial, updated
            print(f"{label} ACCEPT {name}: {change['before_channels']}->{change['after_channels']}", flush=True)
        except (ValueError, RuntimeError, IndexError, KeyError, AssertionError) as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            row.update(status="rejected", reason=f"{type(error).__name__}: {error}")
            print(f"{label} REJECT {name}: {error}", flush=True)
        records.append(row)
        del trial
        gc.collect()
    assert any(r["status"] == "accepted" for r in records)
    backward_check(model, example.device)
    path = work / label / "raw.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, path, base)
    loaded = YOLO(str(path)).model
    assert shape_signature(model) == shape_signature(loaded)
    final_widths = {n: len(v) for n, v in identities.items() if len(v) != initial[n]}
    params, gmac = stats(model, example)
    del model, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return path, dict(actions=records, final_widths=final_widths, parameters=params, gmacs=gmac)


def write_report(run_dir, info):
    rows = info.get("comparisons", [])
    baseline = next((r for r in rows if r["label"] == "baseline"), None)
    chosen = next((r for r in rows if r.get("selected")), None)
    lines = ["# 实验06：敏感度分层 A–E 剪枝对比", "",
             f"运行：`{run_dir.relative_to(ROOT).as_posix()}`；状态：{info['status']}。", "",
             "A–E 是逐档提高剪枝比例；本次另外采用训练数据的梯度重要性来选通道。每组都从实验02同一个 best.pt 开始。",
             "复用实验03的 L1 屏蔽敏感度：低≤0.005，中为(0.005,0.020]，高>0.020（mAP50-95绝对下降）。这只是分组代理，未重新测梯度选通道后的逐层敏感度。", "",
             "| 方案 | 低敏感层 | 中敏感层 | 高敏感层 |", "|---|---:|---:|---:|"]
    for label, ratios in SCHEMES.items():
        lines.append(f"| {label} | {ratios[0]:.1%} | {ratios[1]:.1%} | {ratios[2]:.1%} |")
    lines += ["", "全部方案共用一个脚本。通道数四舍五入为整数，未强制对齐8，以免20%与25%在64通道层上变成相同实验。比例针对候选根层，残差等依赖可能联动其他层；实际宽度和拒绝原因保存在统一run_info.json。",
              "候选根层要求≥64通道，排除首层、注意力、检测头及已知不兼容的model.13.m.0.cv2.conv。保护CSP分块宽度和检测最终输出；依赖允许调整检测头输入和关联深度卷积。每层至少保留原始通道一半。", "",
              "| 对照/方案 | 参数减少 | GMAC减少 | 剪后mAP50-95 | 微调best mAP50-95 | 保留mAP50-95 | mAP50 | 前向ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        name = r['label'] + ("（选中）" if r.get('selected') else "")
        lines.append(f"| {name} | {r.get('parameter_reduction',0):.2%} | {r.get('compute_reduction',0):.2%} | {r['raw']['map50_95']:.4f} | {r['finetuned']['map50_95']:.4f} | {r['map50_95']:.4f} | {r['map50']:.4f} | {r.get('latency_median_ms',0):.3f} |")
    if chosen and baseline:
        drop = baseline['map50_95']-chosen['map50_95']
        lines += ["", f"选择 **{chosen['label']}**：参数减少{chosen['parameter_reduction']:.2%}，GMAC减少{chosen['compute_reduction']:.2%}；mAP50-95从{baseline['map50_95']:.4f}变为{chosen['map50_95']:.4f}（下降{drop*100:.2f}个百分点）。",
                  f"预先固定规则：在下降≤{info['configuration']['max_map_drop']*100:.1f}个百分点的方案中选GMAC减少最多者；无符合者则选精度最高者并明确不达标。本次规则结果：{info['selection_status']}。这只是本次五组中的选择，不保证全局最优。",
                  f"保留检查点来源：{chosen['checkpoint_stage']}。每组统一比较剪后原始模型与微调best，保留两者中mAP50-95较高者；如果微调未提升，不把结果称为精度恢复。"]
    calibration = info.get('calibration', {})
    lines += ["", f"过程：用全部{calibration.get('images',0)}张训练图片（{calibration.get('batches',0)}个batch）统计 mean(|W×∂L/∂W|)，FP32、无增强、逐batch清梯度、不更新权重、BN固定；已检查模型张量未被统计过程改动。原始通道编号随依赖删除同步更新，未退回L1排序。",
              f"微调：每组及未剪枝对照均{info['configuration']['epochs']}轮，AdamW、lr0=0.0001、batch=8、640、seed=42，BN统计与仿射参数固定、关闭强增强。检查损失有限、结构保持及保存后重载。",
              "测速：同一GPU、FP32、batch=1、640×640、融合后网络前向，每模型预热30次，5轮共250次同步计时；不包含读图/预处理/NMS。GMAC按未融合结构统一计算，不等于端到端速度。", "",
              "数据：102张训练、26张验证，同一验证集参与分层、选best及选方案，结果只用于小样本方法筛选。官方预训练也可能见过这些COCO图片；后续需独立数据/COCO2017复核。", "",
              f"耗时：{info.get('elapsed_seconds',0):.1f}秒；GPU峰值：{info.get('peak_allocated_mb',0):.0f}MB。保留一个共享脚本、一个选中模型best.pt、一份汇总run_info.json与完整run.log；不为五组各写一套代码或保留重复训练输出。",
              "", "结构剪枝依赖：[Torch-Pruning / DepGraph](https://github.com/VainF/Torch-Pruning)。", ""]
    text = '\n'.join(lines)
    (run_dir/'report.md').write_text(text, encoding='utf-8')
    (ROOT/'reports/experiment06_gradient_tiered.md').write_text(text, encoding='utf-8')


def run(args, run_dir, info):
    init_seeds(42, deterministic=True)
    base = ROOT/'runs/train/experiment02_coco128/weights/best.pt'
    data = ROOT/'configs/coco128_split.yaml'
    sensitivity = ROOT/'reports/experiment03_sensitivity.csv'
    work = run_dir/'_work'
    work.mkdir()
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    tiers = load_tiers(sensitivity)
    info.update(configuration=vars(args), tiers=tiers,
                sources={str(p.relative_to(ROOT)): sha(p) for p in (base,data,sensitivity,Path(__file__),ROOT/'scripts/prune_greedy.py',ROOT/'scripts/prune_independent_compare.py',ROOT/'scripts/greedy_evaluation.py')},
                environment={k: importlib.metadata.version(k) for k in ('torch','ultralytics','torch-pruning')},
                python=sys.version, platform=platform.platform(), gpu=torch.cuda.get_device_name(device),
                selection_rule='max_compute_saved_if_map_drop_at_most_limit_else_highest_map', status='calibrating')
    write_json(run_dir/'run_info.json', info)
    scores, info['calibration'] = collect_scores(base, data, tiers, device)
    info['status'] = 'pruning'
    info['scheme_details'] = {}
    write_json(run_dir/'run_info.json', info)
    example = torch.zeros(1,3,640,640,device=device)
    source_model = YOLO(str(base)).model.float().to(device).eval()
    baseline_params, baseline_gmacs = stats(source_model, example)
    del source_model
    paths = {'baseline': base}
    for label in SCHEMES:
        paths[label], info['scheme_details'][label] = apply_scheme(label,base,tiers,scores,example,work)
        write_json(run_dir/'run_info.json', info)
    info.update(status='finetuning', comparisons=[])
    selected_paths = {}
    for label, path in paths.items():
        started = time.perf_counter()
        raw = evaluate(path,data,work/'val'/f'{label}_raw',args.device)
        best = run_finetune(path,data,work/'train'/label,args.device,epochs=args.epochs)
        ft = evaluate(best,data,work/'val'/f'{label}_ft',args.device)
        use_ft = ft['map50_95'] > raw['map50_95']+1e-7
        selected_paths[label] = best if use_ft else path
        chosen_metrics = ft if use_ft else raw
        row = dict(label=label, raw=raw, finetuned=ft, **chosen_metrics,
                   checkpoint_stage='finetuned_best' if use_ft else 'raw',
                   seconds=time.perf_counter()-started, selected=False)
        # Keep the learning curve in the one run manifest before cleaning work.
        with (work/'train'/label/'results.csv').open(encoding='utf-8-sig') as handle:
            row['training_curve'] = list(csv.DictReader(handle))
        info['comparisons'].append(row)
        print(f"FINISHED {label}: raw={raw['map50_95']:.6f} finetuned={ft['map50_95']:.6f}", flush=True)
        write_json(run_dir/'run_info.json',info)
        gc.collect()
        torch.cuda.empty_cache()
    info['status'] = 'benchmarking'
    write_json(run_dir/'run_info.json',info)
    performance = benchmark(selected_paths,args.device)
    for row in info['comparisons']:
        row.update(performance[row['label']])
        row['parameter_reduction'] = 1-row['parameters']/baseline_params
        row['compute_reduction'] = 1-row['gmacs']/baseline_gmacs
    baseline = info['comparisons'][0]
    options = info['comparisons'][1:]
    feasible = [r for r in options if baseline['map50_95']-r['map50_95'] <= args.max_map_drop]
    if feasible:
        selected = max(feasible,key=lambda r:(r['compute_reduction'],r['map50_95']))
        info['selection_status'] = '达到精度阈值'
    else:
        selected = max(options,key=lambda r:(r['map50_95'],r['compute_reduction']))
        info['selection_status'] = '五组均未达到精度阈值，仅保留精度最高者作后续起点'
    selected['selected'] = True
    target = run_dir/'best.pt'
    shutil.copy2(selected_paths[selected['label']],target)
    verified = evaluate(target,data,work/'val'/'selected_reload',args.device)
    assert abs(verified['map50_95']-selected['map50_95']) < 1e-6
    assert all(sha(ROOT/p) == digest for p,digest in info['sources'].items())
    info.update(selected_scheme=selected['label'], selected_model=str(target),
                selected_sha256=sha(target), verified_reload=verified,
                peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1e6,
                elapsed_seconds=time.perf_counter()-info.pop('_timer'), finished_at=now(),status='complete')
    write_json(run_dir/'run_info.json',info)
    write_report(run_dir,info)
    # Only remove this invocation's explicitly verified temporary subtree.
    resolved = work.resolve()
    assert resolved.parent == run_dir.resolve() and resolved.name == '_work'
    assert target.is_file() and (run_dir/'report.md').is_file()
    shutil.rmtree(resolved)
    info['temporary_outputs_removed'] = True
    write_json(run_dir/'run_info.json',info)
    print(f"SELECTED {selected['label']} GMAC_saved={selected['compute_reduction']:.2%} mAP={selected['map50_95']:.6f}",flush=True)
    print('COMPLETE '+str(run_dir),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='0')
    parser.add_argument('--epochs',default=10,type=int)
    parser.add_argument('--max-map-drop',default=.02,type=float)
    args = parser.parse_args()
    os.chdir(ROOT)
    run_dir = ROOT/'runs/prune/experiment06_gradient_tiered'/datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True,exist_ok=False)
    info = dict(started_at=now(),status='starting',_timer=time.perf_counter())
    original_out, original_err = sys.stdout, sys.stderr
    with (run_dir/'run.log').open('w',encoding='utf-8',buffering=1) as log:
        sys.stdout, sys.stderr = Tee(original_out,log), Tee(original_err,log)
        handlers = [(h,h.stream) for h in LOGGER.handlers if hasattr(h,'stream')]
        for h,_ in handlers:
            h.setStream(sys.stdout)
        try:
            print('RUN_DIR '+str(run_dir),flush=True)
            run(args,run_dir,info)
        except BaseException as error:
            info.update(status='failed',error=repr(error),finished_at=now())
            if '_timer' in info:
                info['elapsed_seconds'] = time.perf_counter()-info.pop('_timer')
            write_json(run_dir/'run_info.json',info)
            traceback.print_exc()
            raise
        finally:
            for h,stream in handlers:
                h.setStream(stream)
            sys.stdout,sys.stderr = original_out,original_err


if __name__ == '__main__':
    main()
