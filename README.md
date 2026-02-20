# sam3d_metric_scale

RGB-D 입력을 활용해 SAM3D를 안정적으로 구동하고, 실측(또는 추정) 깊이 기반으로 스케일이 보정된 3D 결과를 얻기 위한 로컬 파이프라인입니다.

## GitHub 레포 디스크립션
- Description: Single-image SAM2 → MoGe2 → SAM3D pipeline for metric scale research and visualization.
- Topics: `sam2`, `sam3d`, `moge2`, `metric-depth`, `3d`, `point-cloud`, `gradio`, `sim2real`
- 별도 파일: `REPO_DESCRIPTION.md`

## 구성
- `src/`
  - `image_point.py`: SAM2 포인트 기반 마스크 UI
  - `moge_scale.py`: 마스크 영역 MoGe depth + 스케일 추정
  - `sam3d_export.py`: 이미지+마스크 → SAM3D 결과(.ply)
  - `sam3d_scale.py`: 스케일 알고리즘 테스트 러너(ICP/TEASER++)
  - `sam3d_scale_teaserpp.py`: TEASER++ 기반 스케일 추정
  - `sam3d_scale_utils.py`: 공통 유틸 + 매칭 시각화
- `datas/`: 샘플 이미지
- `outputs/`: 결과 저장(자동 생성, gitignored)
- 외부 레포(로컬 의존, gitignored): `sam2/`, `sam-3d-objects/`, `MoGe/`, (옵션) `TEASER-plusplus/`

## 외부 레포 링크
- SAM2: https://github.com/facebookresearch/sam2
- SAM3D Objects: https://github.com/facebookresearch/sam-3d-objects
- MoGe: https://github.com/microsoft/MoGe
- TEASER++ (옵션): https://github.com/MIT-SPARK/TEASER-plusplus

## 출력 구조
- 기본 루트: `outputs/<image_stem>[_###]/`
  - `sam2_masks/`: SAM2 마스크
  - `moge_scale/`: MoGe 스케일 결과(JSON/NPZ/PLY)
  - `sam3d/`: SAM3D 결과(Ply)
  - `sam3d_scale/`: 스케일 값(txt) + 스케일 적용 PLY + 스케일 메시 + 디케메이트 메시
- 동일 이름 폴더가 있으면 `_001`, `_002`처럼 번호가 붙습니다.
- 원본 이미지는 출력 루트에 복사됩니다.
- 메시 디케메이트 결과는 `*_scaled_mesh_decimated.{glb|ply|obj}`로 저장됩니다.

## 사전 준비
- Conda env: `sam2`, `sam3d-objects`, `moge` (옵션), `teaserpp` (TEASER++ 사용 시에만 필요)
  - `--run-moge`를 주지 않으면 MoGe 환경은 필요 없습니다.
  - 기본 `--scale-algo`는 `icp`라서 TEASER++ 환경 없이도 실행됩니다.
- 외부 레포 위치:
  - `sam2/`, `sam-3d-objects/`, `MoGe/`를 이 레포 루트에 두는 구성을 권장합니다.
  - 다른 위치라면 `SAM2_ROOT`, `SAM3D_ROOT`, `MOGE_ROOT`로 지정하세요.
- SAM3D 사용은 HF 승인 필요.

## 설치 (권장 구성)

### 1) 외부 레포 배치
```bash
git clone https://github.com/facebookresearch/sam2.git sam2
git clone https://github.com/facebookresearch/sam-3d-objects.git sam-3d-objects
git clone https://github.com/microsoft/MoGe.git MoGe
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

### 5) (옵션) TEASER++ 상세 설치
TEASER++는 PyPI에 `teaserpp-python` 휠이 없어서 소스 빌드가 필요합니다.

#### TEASER++ (Python 바인딩, 소스 빌드)
```bash
sudo apt install cmake libeigen3-dev libboost-all-dev

conda create -n teaserpp python=3.10 -y
conda run -n teaserpp python -m pip install -U pip setuptools wheel numpy

# TEASER++ 레포는 이 레포 루트의 TEASER-plusplus/를 사용
conda run -n teaserpp python -m pip install -v ./TEASER-plusplus

# 동작 확인
conda run -n teaserpp python - <<'PY'
import teaserpp_python
print("teaserpp_python ok")
PY
```
- `pip install teaserpp-python`는 현재 사용 가능한 휠이 없어 실패합니다.
- PCL/Open3D는 예제용이며, 본 프로젝트의 스케일 추정에는 필요하지 않습니다.

## 사용법

### 1) 전체 파이프라인 실행 (기본)
```bash
./run_full_pipeline.sh \
  --image /path/to/rgb.png \
  --depth-image /path/to/depth.png \
  --cam-k /path/to/cam_K.txt \
  --mesh-target-faces 20000 \
  --output-base outputs/demo
```
필요 시 depth 스케일 추가:
```bash
./run_full_pipeline.sh \
  --image /path/to/rgb.png \
  --depth-image /path/to/depth.png \
  --cam-k /path/to/cam_K.txt \
  --depth-scale 0.001 \
  --output-base outputs/demo
```
MoGe(옵션) 활성화:
```bash
./run_full_pipeline.sh \
  --image /path/to/rgb.png \
  --run-moge \
  --output-base outputs/demo
```

목표 face 수로 직접 지정:
```bash
./run_full_pipeline.sh \
  --image /path/to/rgb.png \
  --run-moge \
  --mesh-target-faces 200000 \
  --output-base outputs/demo
```

비활성화:
```bash
./run_full_pipeline.sh \
  --image /path/to/rgb.png \
  --no-mesh-decimate \
  --output-base outputs/demo
```

자주 쓰는 옵션 요약:
- `--image`: 입력 RGB 이미지
- `--depth-image`: real depth 이미지(있으면 real_scale 생성)
- `--cam-k`: 3x3 intrinsics 텍스트 파일
- `--depth-scale`: depth 스케일 (`auto` 기본, mm → m이면 0.001)
- `--output-base`: 결과 저장 루트
- `--run-moge`: MoGe 실행(기본 off)
- `--scale-algo`: `icp` | `teaserpp` (기본: `icp`)
- `--fine-registration`: `scale_only` 후 ICP로 R/t 추가 정합
- `--mesh-decimate-ratio`: 스케일 보정된 메시의 face 비율 (기본: 0.02)
- `--mesh-target-faces`: 목표 face 수 (비율 대신 사용, 기본: 20000)
- `--no-mesh-decimate`: 메시 밀도 조정 비활성화

### 2) 스케일 알고리즘 단독 실행
```bash
conda run -n sam3d-objects python src/sam3d_scale.py \
  --sam3d-ply /path/to/sam3d.ply \
  --moge-npz /path/to/moge_output.npz \
  --algo icp
```
TEASER++ 예시:
```bash
conda run -n teaserpp python src/sam3d_scale.py \
  --sam3d-ply /path/to/sam3d.ply \
  --moge-npz /path/to/moge_output.npz \
  --algo teaserpp \
  --teaser-estimate-scaling
```

### 3) 메시 밀도 조정 단독 실행
```bash
conda run -n sam3d-objects python src/mesh_decimate.py \
  --input /path/to/sam3d_scale/my_obj_scaled_mesh.ply \
  --target-faces 20000
```
옵션 예시:
- `--target-faces 200000`: 목표 face 수 지정
- `--method open3d`: open3d quadric decimation 강제

## 스케일 추정 방식
- `sam3d_scale.py`에서 알고리즘을 선택해 실험합니다.
  - `icp`: Umeyama + ICP
  - `teaserpp`: TEASER++ (외부 의존)
- 산출물: `*_scale.txt`에 아래 3줄 형식으로 저장합니다.
  - `base`: SAM3D pose scale (벡터)
  - `extra`: 정합으로 추정한 추가 스케일 (스칼라)
  - `final`: 최종 적용 스케일(벡터, `base * extra`)

## 환경 변수
- `SAM2_ROOT`: SAM2 레포 경로
- `SAM3D_ROOT`: SAM3D Objects 레포 경로
- `MOGE_ROOT`: MoGe 레포 경로

## 시각화 의존성
- 파이프라인은 기본적으로 시각화를 포함하지 않습니다. 필요 시 별도 도구로 출력 결과를 확인하세요.

## 트러블슈팅
- `ModuleNotFoundError: sam2`  
  - `conda run -n sam2 python -m pip install -e ./sam2` 재설치 또는 `SAM2_ROOT` 설정.
- `ModuleNotFoundError: moge`  
  - `conda run -n moge python -m pip install -e ./MoGe` 재설치 또는 `MOGE_ROOT` 설정.
- `ModuleNotFoundError: sam3d_objects`  
  - `conda run -n sam3d-objects python -m pip install -e ".[inference]"` 재설치 또는 `SAM3D_ROOT` 설정.
- `pip install teaserpp-python` 실패  
  - PyPI에 해당 패키지 휠이 없어 발생합니다. `TEASER-plusplus/`에서 `pip install -v .`로 소스 빌드를 진행하세요.
- `ImportError: teaserpp_python`  
  - `teaserpp` 환경에서 `pip install -v ./TEASER-plusplus`가 정상 완료되었는지 확인하세요.
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
