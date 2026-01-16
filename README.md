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
  - `sam3d_scale.py`: 스케일 알고리즘 테스트 러너(ICP/RANSAC/RMS/TEASER++/Super4PCS)
  - `sam3d_scale_icp.py`: Umeyama + ICP 스케일 추정
  - `sam3d_scale_ransac.py`: Umeyama + RANSAC 스케일 추정
  - `sam3d_scale_rms.py`: RMS 비율 기반 스케일 추정
  - `sam3d_scale_teaserpp.py`: TEASER++ 기반 스케일 추정
  - `sam3d_scale_super4pcs.py`: Super4PCS 기반 스케일 추정
  - `sam3d_scale_utils.py`: 공통 유틸 + 매칭 시각화
  - `visualize_outputs.py`: 결과 폴더 기반 통합 시각화
- `run_scale_test.sh`: 스케일 알고리즘 테스트 실행 스크립트
- `datas/`: 샘플 이미지
- `outputs/`: 결과 저장(자동 생성, gitignored)
- 외부 레포(로컬 의존, gitignored): `sam2/`, `sam-3d-objects/`, `MoGe/`, (옵션) `TEASER-plusplus/`, `Super4PCS/`

## 출력 구조
- 기본 루트: `outputs/<image_stem>[_###]/`
  - `sam2_masks/`: SAM2 마스크
  - `moge_scale/`: MoGe 스케일 결과(JSON/NPZ/PLY)
  - `sam3d/`: SAM3D 결과(Ply)
  - `sam3d_scale/`: 스케일 값(txt) + 스케일 적용 PLY(선택 실행)
- 동일 이름 폴더가 있으면 `_001`, `_002`처럼 번호가 붙습니다.
- 원본 이미지는 출력 루트에 복사됩니다.

## 사전 준비
- Conda env: `sam2`, `sam3d-objects`, `moge` (옵션) `teaserpp`
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

### 5) (옵션) TEASER++ / Super4PCS 상세 설치
TEASER++는 PyPI에 `teaserpp-python` 휠이 없어서 소스 빌드가 필요합니다.
Super4PCS는 CMake로 바이너리를 빌드해서 사용합니다.

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

#### Super4PCS (바이너리 빌드)
Super4PCS는 현재 유지보수 종료(deprecated) 상태이며, 문서에서는 OpenGR 사용을 권장합니다.
본 프로젝트는 기존 Super4PCS 바이너리를 그대로 호출합니다.
```bash
git clone https://github.com/nmellado/Super4PCS Super4PCS
cmake -S Super4PCS -B Super4PCS/build -DCMAKE_BUILD_TYPE=Release
cmake --build Super4PCS/build -j

# 바이너리 위치 확인 후 환경변수로 등록
find Super4PCS/build -type f -name Super4PCS
# 일반적으로 아래 경로에 생성됩니다.
# Super4PCS/build/demos/Super4PCS/Super4PCS
export SUPER4PCS_BIN="/abs/path/to/Super4PCS"
```
- 첫 빌드 시 Eigen 등 일부 의존성은 CMake가 자동 다운로드할 수 있어 네트워크가 필요합니다.
- `SUPER4PCS_BIN` 또는 `SUPER4PCS_ROOT`를 설정하면 `run_scale_test.sh`가 자동으로 바이너리를 찾습니다.

`run_scale_test.sh`에서 TEASER++는 `--sam3d-env teaserpp`, Super4PCS는 `--super4pcs-bin` 또는 `SUPER4PCS_BIN`으로 실행합니다.

## 사용법

### 1) 스케일 알고리즘 테스트
```bash
./run_scale_test.sh \
  --output-root outputs/coffee_maker_sample \
  --algo icp \
  --show-viz
```
개별 파일 지정:
```bash
./run_scale_test.sh \
  --base_root /home/vision/Sim2Real_Data_Augmentation_for_VLA/sam3d_metric_scale/outputs/coffee_maker_sample \
  --moge_file /moge_scale/coffee_maker_sample_coffee_maker_sample_000.npz \
  --sam3d_file /sam3d/coffee_maker_sample_000.ply \
  --algo ransac
```
`--moge_file`, `--sam3d_file`는 `--base_root` 기준 상대 경로로 처리됩니다.
조절 가능한 주요 옵션:
- 공통: `--max-points`, `--seed`
- ICP: `--icp-max-iters`, `--icp-tolerance`, `--icp-nn-max-points`, `--icp-trim-ratio`
- RANSAC: `--ransac-iters`, `--ransac-sample`, `--ransac-inlier-thresh`, `--ransac-nn-max-points`
- RMS: `--rms-nn-max-points`
- TEASER++: `--teaser-noise-bound`, `--teaser-nn-max-points`, `--teaser-max-correspondences`, `--teaser-gnc-factor`, `--teaser-rot-max-iters`, `--teaser-cbar2`, `--teaser-estimate-scaling`
- Super4PCS: `--super4pcs-bin`, `--super4pcs-overlap`, `--super4pcs-delta`, `--super4pcs-timeout`
- 시각화: `--show-viz`, `--save-viz`, `--viz-max-points`, `--viz-max-pairs`
여러 마스크 일괄 실행:
```bash
./run_scale_test.sh \
  --output-root outputs/coffee_maker_sample \
  --all \
  --algo ransac \
  --save-viz
```
TEASER++ 예시:
```bash
./run_scale_test.sh \
  --output-root outputs/coffee_maker_sample \
  --algo teaserpp \
  --teaser-estimate-scaling \
  --sam3d-env teaserpp
```
Super4PCS 예시:
```bash
./run_scale_test.sh \
  --output-root outputs/coffee_maker_sample \
  --algo super4pcs \
  --super4pcs-bin /path/to/Super4PCS
```
Super4PCS 파라미터 가이드(원본 Usage 요약):
- `--super4pcs-overlap`: 두 포인트클라우드의 예상 겹침 비율(0~1). 알 수 없으면 1.0에서 시작해 점차 낮추며 찾는 방식을 권장.
- `--super4pcs-delta`: 정합 허용 오차(장면 단위). 너무 작으면 실패, 너무 크면 정확도 저하.
- `--super4pcs-timeout`: 랜덤 탐색 시간(초). 난이도 높은 경우 `100`~`1000` 등 크게 주는 것이 안정적.
```bash
conda run -n sam3d-objects python src/sam3d_scale.py \
  --sam3d-ply /path/to/sam3d.ply \
  --moge-npz /path/to/moge_output.npz \
  --algo icp \
  --show-viz
```
다른 알고리즘 예시:
```bash
conda run -n sam3d-objects python src/sam3d_scale.py \
  --sam3d-ply /path/to/sam3d.ply \
  --moge-npz /path/to/moge_output.npz \
  --algo ransac --ransac-iters 300 --ransac-sample 64 --ransac-inlier-thresh 0.02 \
  --save-viz
```
```bash
conda run -n sam3d-objects python src/sam3d_scale.py \
  --sam3d-ply /path/to/sam3d.ply \
  --moge-npz /path/to/moge_output.npz \
  --algo rms
```

## 스케일 추정 방식
- `sam3d_scale.py`에서 알고리즘을 선택해 실험합니다.
  - `icp`: Umeyama + ICP
  - `ransac`: Umeyama + RANSAC
  - `rms`: RMS 비율 기준선
  - `teaserpp`: TEASER++ (외부 의존)
  - `super4pcs`: Super4PCS (외부 바이너리)
- 산출물: `*_scale.txt`에 스케일 값만 저장합니다.

## 환경 변수
- `SAM2_ROOT`: SAM2 레포 경로
- `SAM3D_ROOT`: SAM3D Objects 레포 경로
- `MOGE_ROOT`: MoGe 레포 경로
- `SUPER4PCS_BIN`: Super4PCS 바이너리 경로
- `SUPER4PCS_ROOT`: Super4PCS 레포 경로

## 시각화 의존성
- 스케일 매칭 시각화(`sam3d_scale.py --show-viz`)는 `matplotlib`가 필요합니다.
```bash
conda run -n sam3d-objects python -m pip install matplotlib
```

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
