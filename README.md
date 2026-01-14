# sam3d_metric_scale

단일 이미지에서 SAM2 마스크 → MoGe2 metric depth → SAM3D 3D 결과를 생성하고, 스케일 보정 연구를 위한 파이프라인을 제공하는 로컬 레포입니다.

## GitHub 레포 디스크립션
- Description: Single-image SAM2 → MoGe2 → SAM3D pipeline for metric scale research and visualization.
- Topics: `sam2`, `sam3d`, `moge2`, `metric-depth`, `3d`, `point-cloud`, `gradio`, `sim2real`
- 별도 파일: `REPO_DESCRIPTION.md`

## 구성
- `src/`
  - `image_point.py`: SAM2 포인트 기반 마스크 UI
  - `moge_scale.py`: 마스크 영역 MoGe depth + 스케일 추정
  - `sam3d_export.py`: 이미지+마스크 → SAM3D 결과(.ply)
  - `visualize_outputs.py`: 결과 폴더 기반 통합 시각화
- `run_full_pipeline.sh`: SAM2 → MoGe → SAM3D 통합 파이프라인
- `run_visualize_outputs.sh`: 결과 시각화 실행 스크립트
- `datas/`: 샘플 이미지
- `outputs/`: 결과 저장(자동 생성, gitignored)
- 외부 레포(로컬 의존, gitignored): `sam2/`, `sam-3d-objects/`, `MoGe/`

## 출력 구조
- 기본 루트: `outputs/<image_stem>[_###]/`
  - `sam2_masks/`: SAM2 마스크
  - `moge_scale/`: MoGe 스케일 결과(JSON/NPZ)
  - `sam3d/`: SAM3D 결과(Ply)
- 동일 이름 폴더가 있으면 `_001`, `_002`처럼 번호가 붙습니다.
- 원본 이미지는 출력 루트에 복사됩니다.

## 사전 준비
- Conda env: `sam2`, `sam3d-objects`, `moge`
- 외부 레포 위치:
  - `sam2/`, `sam-3d-objects/`, `MoGe/`를 이 레포 루트에 두는 구성을 권장합니다.
  - 다른 위치라면 `SAM2_ROOT`, `SAM3D_ROOT`, `MOGE_ROOT`로 지정하세요.
- SAM3D 사용은 HF 승인 필요.

## 설치 (권장 구성)

### 1) 외부 레포 배치
```bash
git clone <SAM2_REPO_URL> sam2
git clone <SAM3D_OBJECTS_REPO_URL> sam-3d-objects
git clone <MOGE_REPO_URL> MoGe
```

### 2) SAM2 환경
```bash
conda create -n sam2 python=3.10 -y
conda run -n sam2 python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n sam2 python -m pip install -e ./sam2
conda run -n sam2 python -m pip install opencv-python
bash sam2/checkpoints/download_ckpts.sh
```

### 3) SAM3D Objects 환경
```bash
conda env create -f sam-3d-objects/environments/default.yml
PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
  conda run -n sam3d-objects python -m pip install -e "./sam-3d-objects[dev]"
PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
  conda run -n sam3d-objects python -m pip install -e "./sam-3d-objects[p3d]"
PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
  conda run -n sam3d-objects python -m pip install -e "./sam-3d-objects[inference]"
conda run -n sam3d-objects python sam-3d-objects/patching/hydra
conda run -n sam3d-objects hf auth login
```
체크포인트 다운로드(예시):
```bash
conda run -n sam3d-objects hf download --repo-type model \
  --local-dir sam-3d-objects/checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf
```

### 4) MoGe 환경
```bash
conda create -n moge python=3.10 -y
conda run -n moge python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
conda run -n moge python -m pip install -r MoGe/requirements.txt
conda run -n moge python -m pip install -e ./MoGe
```

## 사용법

### 1) 통합 파이프라인 실행
```bash
./run_full_pipeline.sh \
  --image datas/coffee_maker_sample.jpg
```
UI에서 마스크 저장 후 `q`로 종료하면 MoGe → SAM3D 순서로 실행됩니다.

### 2) 결과 시각화
```bash
./run_visualize_outputs.sh \
  --output-root outputs/coffee_maker_sample
```
- Gradio UI에서 이미지/마스크/MoGe depth/MoGe 포인트클라우드/SAM3D PLY를 확인합니다.
- MoGe 포인트클라우드는 축이 포함된 3D 플롯으로 표시됩니다(파일 저장 없음).

### 3) 개별 실행(옵션)
```bash
conda run -n sam3d-objects python src/sam3d_export.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png
```
```bash
conda run -n moge python src/moge_scale.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png
```

## 환경 변수
- `SAM2_ROOT`: SAM2 레포 경로
- `SAM3D_ROOT`: SAM3D Objects 레포 경로
- `MOGE_ROOT`: MoGe 레포 경로

## 시각화 의존성
- Gradio UI는 `sam3d-objects` 환경에서 동작합니다.
- MoGe 포인트클라우드 플롯은 `plotly`가 있으면 3D로 표시됩니다. 없으면 `matplotlib`로 fallback 됩니다.
```bash
conda run -n sam3d-objects python -m pip install gradio plotly matplotlib
```

## 트러블슈팅
- `ModuleNotFoundError: sam2`  
  - `conda run -n sam2 python -m pip install -e ./sam2` 재설치 또는 `SAM2_ROOT` 설정.
- `ModuleNotFoundError: moge`  
  - `conda run -n moge python -m pip install -e ./MoGe` 재설치 또는 `MOGE_ROOT` 설정.
- `ModuleNotFoundError: sam3d_objects`  
  - `conda run -n sam3d-objects python -m pip install -e ".[inference]"` 재설치 또는 `SAM3D_ROOT` 설정.
- SAM3D 메모리 이슈  
  - VRAM 32GB 권장. 24GB에서는 추론 실패 가능.

## .gitignore 안내
- `outputs/`, `hugging_face_token.txt`, `sam2/`, `sam-3d-objects/`, `MoGe/`는 기본적으로 gitignored입니다.
- 외부 레포는 submodule로 전환하거나, 별도 경로에서 관리하세요.

## 라이센스
- 이 레포의 코드: `LICENSE` 참고.
- 외부 레포(SAM2, SAM3D Objects, MoGe)는 각각의 라이센스를 따릅니다.

## Acknowledgements
- SAM2
- SAM3D Objects
- MoGe2
