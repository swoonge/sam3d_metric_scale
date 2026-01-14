# sam3d_metric_scale

SAM3D + MoGe2 스케일 보정 연구를 위한 로컬 작업 공간입니다.

## 목적/범위
- SAM3D 결과를 MoGe2의 metric depth로 보정하는 프로토타이핑.
- 실험용 스크립트와 실행 로그를 이 레포에서 관리.

## 구성
- `image_point.py`: SAM2 포인트 기반 마스크 UI
- `sam3d_export.py`: 이미지+마스크 → SAM3D 결과(.ply)
- `moge_scale.py`: 마스크 영역의 MoGe depth + 스케일 추정
- `run_full_pipeline.sh`: SAM2 UI → MoGe → SAM3D 통합 파이프라인
- `visualize_outputs.py`: 결과 폴더 기반 통합 시각화
- `run_visualize_outputs.sh`: 시각화 실행용 셸
- `datas/`: 샘플 이미지
- `outputs/`: 결과 저장(자동 생성, gitignored)

## 출력 경로
- 기본 루트: `outputs/<image_stem>[_###]/`
  - `sam2_masks/`: SAM2 마스크
  - `moge_scale/`: MoGe 스케일 결과(JSON/NPZ)
  - `sam3d/`: SAM3D 결과(Ply)
- 동일 이름 폴더가 있으면 `_001`, `_002`처럼 번호가 붙습니다.
- 원본 이미지는 출력 루트에 복사됩니다.

## 사전 준비
- Conda env: `sam2`, `sam3d-objects`, `moge`
- 외부 레포 위치:
  - `sam2/`, `sam-3d-objects/`를 이 레포 루트 또는 상위 경로에 두면 기본 경로로 동작합니다.
  - 다른 위치라면 `--image`, `--sam3d-config`로 직접 지정하세요.
- 환경 변수(선택): `SAM2_ROOT`, `SAM3D_ROOT`를 지정하면 개별 실행 시 우선 사용합니다.
- SAM3D 사용은 HF 승인 필요.

## 빠른 시작

### 1) 통합 파이프라인
```bash
./run_full_pipeline.sh \
  --image datas/coffee_maker_sample.jpg
```

### 2) 결과 시각화
```bash
./run_visualize_outputs.sh \
  --output-root outputs/coffee_maker_sample
```
- Gradio UI에서 이미지/마스크/MoGe depth/MoGe 포인트클라우드/SAM3D PLY를 확인합니다.
- MoGe 포인트클라우드는 축이 포함된 3D 플롯으로 표시됩니다(파일 저장 없음).

## 옵션 요약

### run_full_pipeline.sh
- `--image PATH`
- `--output-base PATH`
- `--latest`
- `--sam2-env NAME`
- `--sam3d-env NAME`
- `--sam3d-config PATH`
- `--sam3d-seed INT`
- `--sam3d-compile`
- `--moge-env NAME`
- `--moge-model NAME`
- `--scale-method NAME`
- `--min-pixels INT`

### run_visualize_outputs.sh
- `--output-root PATH`
- `--image PATH`
- `--server-port INT`
- `--share`
- `--no-browser`
- `--moge-max-points N`
- `--moge-axis-fraction F`
- `--moge-axis-steps N`

## 개별 실행(옵션)

### SAM3D export
```bash
conda run -n sam3d-objects python sam3d_export.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png
```

### MoGe scale
```bash
conda run -n moge python moge_scale.py \
  --image /path/to/image.jpg \
  --mask /path/to/mask.png
```
