#!/usr/bin/env python3
"""
Generic NeRF Pipeline Runner – *Batch, No‑CLI*
──────────────────────────────────────────────
* **RUN_ORDER** 리스트에 원하는 장면 키를 넣으면 순차적으로 학습 → 렌더링 → 평가를 진행합니다.
* 각 장면별 파라미터는 CONFIG 딕셔너리에 정의하며, CLI 인자는 전혀 사용하지 않습니다.
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict

# ─────────────────────────────────────
# 장면별 설정
# ─────────────────────────────────────
# 각 항목: source(필수), model(필수), scene, resolution, iterations 등 필요 값 지정
CONFIG: Dict[str, Dict] = {
    "sacre": {
        "source": "datasets/phototourism/sacre-coeur/dense",
        "model": "outputs/sacre/full",
        "scene": "sacre",
        "resolution": 2,
        "iterations": 70_000,
    },
    "trevi": {
        "source": "datasets/phototourism/trevi-fountain/dense",
        "model": "outputs/trevi/full",
        "scene": "trevi",
        "resolution": 2,
        "iterations": 70_000,
    },
    "@@custom": { # @@: 같아야함
        "source": "/path/to/your/dense",
        "model": "outputs/@@custom/full",
        "scene": "@@custom",
        "resolution": 2,
        "iterations": 70_000,
    }, 
    "sculpture": {
        "source": "datasets/dronesplat-dataset/Sculpture/dense",
        "model": "outputs/sculpture/full",
        "scene": "sculpture",
        "resolution": 2,
        "iterations": 70_000,
    },
    "simingshan": {
        "source": "datasets/dronesplat-dataset/Simingshan/dense",
        "model": "outputs/simingshan/full",
        "scene": "simingshan",
        "resolution": 2,
        "iterations": 70_000,
    },
}

# 실행 순서 지정 (CONFIG에 있는 키만 기입)
RUN_ORDER = [
    "sculpture",
    "simingshan"
]

# 사용할 GPU 번호 (쉼표로 복수 지정 가능)
GPU = "5"

# ─────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────

def sh(cmd: str):
    cmd_full = f"CUDA_VISIBLE_DEVICES={GPU} {cmd}"
    print(f"\n$ {cmd_full}")
    subprocess.run(cmd_full, shell=True, check=True)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────
# 파이프라인 함수
# ─────────────────────────────────────

def run_scene(key: str):
    if key not in CONFIG:
        print(f"✖ Unknown key '{key}', skipping.")
        return

    p = CONFIG[key]
    ensure_dir(p["model"])

    print(f"\n=== ▶ Processing '{key}' → {p['model']} ===")

    # 1. Training
    sh(
        "python ./train.py "
        f"--source_path {p['source']} "
        f"--scene_name {p['scene']} "
        f"--model_path {p['model']} "
        f"--eval --resolution {p['resolution']} --iterations {p['iterations']}"
    )

    # 2. Render train & test
    sh(
        f"python ./render.py --model_path {p['model']}"
    )

    # 3. Metrics (right‑half)
    sh(
        f"python ./metrics_half.py --model_path {p['model']}"
    )

    print("✔️  Finished", key)


# ─────────────────────────────────────
# 메인
# ─────────────────────────────────────

def main():
    for key in RUN_ORDER:
        try:
            run_scene(key)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Error while processing '{key}' (exit code {e.returncode}). Continuing to next.")


if __name__ == "__main__":
    main()
