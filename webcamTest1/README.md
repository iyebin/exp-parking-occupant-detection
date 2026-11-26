# ExitGuard 카메라 사전 테스트 — Razer Kiyo Pro + Raspberry Pi 4

YOLOv8n과 YOLO26n이 Pi 4 CPU에서 사람을 얼마나 빠르고 정확하게, 얼마나 멀리서 찾는지 측정합니다.
측정 항목: 모델 크기, 추론 속도, 단계별 병목, 정확도·인원 카운트, 온도, 검출 거리, 지연.

**원칙: 한 번 녹화하고, 모든 설정에 같은 영상을 다시 돌립니다.** 정확도·거리·속도·병목은 녹화 영상으로(오프라인),
지연과 발열만 카메라를 켜고(라이브) 잽니다. 그래야 모델끼리 공정하게 비교됩니다.

---

## 1. 준비물

| 장비 | 용도 | 비고 |
|---|---|---|
| Raspberry Pi 4B (RAM 4 GB 이상 권장) | 측정 대상 | Raspberry Pi OS **64-bit** |
| 공식 5V 3A USB-C 전원 | 전원 | 약한 어댑터는 저전압 → 속도 저하 |
| 팬(+방열판) | 냉각 | 팬 있음/없음, 케이스 열림/닫힘을 기록 |
| Razer Kiyo Pro (RZ19-0364) | 카메라 | **파란색 USB 3.0 포트**에 연결 |
| 삼각대, 줄자, 바닥 테이프 | 거리 표시 | 0.5~8 m 표시 |
| 3색 LED 모듈(KS0310) | 지연 측정(선택) | 빨강 핀 → GPIO17 |
| 노트북 | SSH, 라벨링, 결과 보기 | 아래 3절 |

## 2. Pi 환경 설정 (처음 한 번)

```bash
# 이 폴더를 Pi로 복사한 뒤
cd exitguard_camtest
bash setup_pi.sh            # apt 패키지, 가상환경(~/exitguard-venv), pip 패키지, 카메라 확인
sudo reboot                 # video 그룹 적용
```

테스트할 때마다:

```bash
source ~/exitguard-venv/bin/activate
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # CPU 최고 속도 고정
vcgencmd get_throttled      # throttled=0x0 이어야 정상 (저전압·과열 기록 없음)
```

**카메라 주의 (Kiyo Pro + Linux)**
- 리눅스 커널에 이 카메라 전용 보정(USB 절전 끄기, NO_LPM)이 들어가 있습니다. 카메라가 자꾸 끊기면
  `/boot/firmware/cmdline.txt` 끝에 `usbcore.quirks=1532:0e05:k` 를 추가하고 재부팅하세요.
- `v4l2-ctl`로 설정(특히 초점)을 바꾸다 카메라가 멈췄다는 보고가 있습니다. **한 번에 하나씩** 바꾸고, 멈추면 뽑았다 다시 꽂으세요.
- 화각(103°/90°/80°)은 Windows의 Razer Synapse에서 바꾸는 기능입니다. 리눅스에서 바뀌는지는
  `results/camera/v4l2_controls.txt`에서 zoom 계열 항목을 확인하세요. 안 되면 기본 화각을 줄자로 확인해 기록합니다.

## 3. 노트북 환경

| 할 일 | 필요한 것 |
|---|---|
| Pi 원격 접속 | SSH (VS Code Remote-SSH 추천) |
| 라벨링 | CVAT(app.cvat.ai), Roboflow, Label Studio 중 하나 — 클래스 `person` 하나, YOLO 형식으로 내보내기 |
| 결과 보기 | 엑셀/구글 시트, 또는 `python 09_summarize.py` |
| (선택) 1080p 녹화를 노트북에서 | Python 3.10+, `pip install -r requirements.txt`, `python 02_record.py --device 0 ...` |
| (선택) 카메라 펌웨어·화각 | Windows + Razer Synapse |
| (선택) 스크립트 미리 돌려보기 | 영상 파일로 실행 가능: `python 07_live.py --device clip.avi ...` |

## 4. 폴더 구조

```text
exitguard_camtest/
  common.py              공통 함수(모델 경로, 카메라 열기, 온도 읽기, CSV)
  00_camera_check.py     카메라만: 형식, 캡처 속도, 끊김
  01_export_models.py    모델 내려받기 + ONNX/NCNN 변환 + 모델 크기
  02_record.py           시나리오 녹화
  03_extract_frames.py   라벨링할 프레임 추출
  04_benchmark.py        속도·병목 (오프라인)
  05_accuracy.py         정확도·인원 카운트 (오프라인)
  06_distance.py         검출 거리 (오프라인)
  07_live.py             라이브: 지연, 전체 병목, 발열(장시간)
  08_led_latency.py      LED로 실제 카메라 지연 (선택)
  09_summarize.py        최종 결과표 + 온도 그래프
  models/  recordings/  labeled/  results/      (실행하면 생김)
```

## 5. 실행 순서

| 순서 | 명령 | 결과 파일 | 측정 항목 |
|---|---|---|---|
| 1 | `python 00_camera_check.py` | `results/camera/` | 카메라 단독 FPS, 읽기·색 변환 시간, 끊김 |
| 1+ | `python 00_camera_check.py --modes 1280x720@30 --duration 600` | 〃 | 10분 연속 안정성 (+ `dmesg \| grep -i uvc`) |
| 2 | `python 01_export_models.py` | `models/`, `results/models.csv` | **모델 크기(MB)**, 파라미터, GFLOPs |
| 3 | `python 02_record.py --name ... --seconds ...` (시나리오마다) | `recordings/*.avi` | — |
| 4 | `python 04_benchmark.py` | `results/bench.csv` | **추론 속도(ms)**, **병목(ms)**, CPU·메모리 |
| 5 | `python 03_extract_frames.py --per-clip 20` → 라벨링 → `python 05_accuracy.py` | `results/accuracy*.csv` | **정확도**, 인원 카운트, 빈 화면 오탐 |
| 6 | `python 06_distance.py` | `results/distance*.csv` | **검출 거리** |
| 7 | `python 07_live.py ... --duration 60` | `results/live/` | **지연**, 카메라 포함 전체 병목 |
| 8 | `python 07_live.py ... --duration 900 --cooling fan --case open --ambient 22` | `results/live/` | **온도**(15분), 스로틀링, FPS 변화 |
| 9 | (선택) `python 08_led_latency.py --pick-roi` → `--roi ...` | `results/led_latency.csv` | 실제 카메라 지연 |
| 10 | `python 09_summarize.py` | `results/final_table.*`, 온도 그래프 | 전체 표 |

### 비교하는 설정 (총 24개)

| 모델 | 형식 | 입력 크기 |
|---|---|---|
| YOLOv8n | PyTorch(`pt`), ONNX, NCNN | 320, 416, 640 |
| YOLO26n | PyTorch, ONNX, NCNN + **NMS 없는 모드**(`pt_e2e`, `onnx_e2e`) | 320, 416, 640 |

- 형식만 바꾸는 변환(양자화 없음, FP32)이라 정확도는 같아야 합니다. 다르면 변환 문제입니다.
- YOLO26n은 기본값이 NMS(겹친 박스 정리)를 쓰는 모드입니다. `_e2e`는 NMS 없이 바로 결과를 내는 모드로,
  `nms=False`로 실행합니다. NCNN 변환은 이 모드를 지원하지 않아 PyTorch·ONNX에서만 비교합니다.
- 일부만 돌리려면: `--models yolo26n --formats ncnn --sizes 320 640`

### 녹화 시나리오와 파일 이름 (중요: 이름 규칙대로)

카메라는 입구 RGB 위치처럼 **높이 약 2 m, 아래로 비스듬히** 설치합니다. 1080p로 녹화합니다.

| 시나리오 | 이름 예 | 내용 | 길이 |
|---|---|---|---|
| S1 빈 방 | `S1_empty` | 사람 없음, 의자·가방만 | 10 s |
| S2 거리 | `S2_d0.5_center` … `S2_d8.0_edge` | 1명이 바닥 표시 0.5/1/1.5/2/3/4/5/6/8 m에, 화면 가운데(`center`)와 가장자리(`edge`). **다른 사람은 화면 밖** | 각 5 s |
| S3 여러 명 | `S3_p2_side`, `S3_p4_overlap` | 2~5명, 나란히·겹쳐서 | 10 s |
| S4 걷기 | `S4_walk_01` | 화면을 가로질러 나감 | 15 s |
| S5 자세 | `S5_sit`, `S5_crouch`, `S5_cut` | 앉기, 쪼그리기, 화면에 반만 | 10 s |
| S6 조명 | `S6_dim_p2` | 조명 절반 끄고 2명 | 10 s |

```bash
python 02_record.py --name S2_d3.0_center --seconds 5 --note "형광등 전부 켬"
```

Pi에서 1080p 녹화 시 프레임이 빠지면(`frames_dropped`가 0이 아니면) 같은 카메라 설정으로 노트북에서 녹화해 `recordings/`에 복사하세요.

### 라벨링

1. `python 03_extract_frames.py --per-clip 20` → `labeled/images/`
2. CVAT 등에서 `person` 한 클래스로 박스를 그림. **사람이 없는 이미지도 빈 라벨로** 내보내야 오탐을 셀 수 있습니다.
3. YOLO 형식 txt를 `labeled/labels/`에 (이미지와 같은 파일 이름)

## 6. 결과 읽는 법

`results/final_table.csv` 한 줄 = 모델 × 형식 × 입력 크기.

| 열 | 의미 |
|---|---|
| `size_mb` | 모델 파일 크기 |
| `inf_ms`, `total_ms`, `total_ms_p95`, `fps` | 모델 계산 시간, 한 장 전체 처리 시간(중앙값, 느린 5%), 초당 처리 수 |
| `bottleneck` | 가장 오래 걸린 단계 (preprocess / inference / postprocess / count / other) |
| `recall`, `precision`, `ap50` | 사람을 놓치지 않는 비율, 오검출이 적은 비율, 종합 점수 |
| `count_exact_rate`, `undercount_rate` | 인원 수가 정확한 비율, **적게 센 비율(위험한 쪽)** |
| `empty_false_alarm_rate` | 빈 방에서 사람이 있다고 한 비율 |
| `max_dist_center_m`, `max_dist_edge_m` | 검출률 90% 이상인 가장 먼 거리 (NCNN으로 측정, 형식과 무관) |
| `live_fps`, `live_e2e_ms_p95` | 카메라 포함 실제 속도, 프레임 도착→결과까지 지연(느린 5%) |
| `temp_max_c`, `first_throttle_s` | 최고 온도, 스로틀링이 처음 일어난 시점 |

**통과 기준(제안, 팀에서 확정)**
- 속도: NoIR 역할은 한 장 0.5 s 이하, 입구 역할은 추적 포함 5 fps 이상
- 정확도: 정한 신뢰도 기준에서 recall 0.95 이상, 적게 센 비율 최소
- 거리: 승강기 방에서 가장 먼 거리(약 5 m)에서 검출률 90% 이상
- 온도: 실제 쓸 냉각 조건에서 15분 동안 80°C 미만

**안전:** `07_live.py`는 80°C가 10초 이어지거나 저전압이 감지되면 스스로 멈춥니다. Pi 4는 80°C부터 속도를 줄이고 85°C부터 크게 줄입니다.

## 7. 알아둘 점

- `04~06`은 `--source-res 1280x720`으로 라이브 캡처 크기에 맞춰 줄인 프레임을 씁니다. 1080p로 돌리려면 `--source-res none`.
- `04_benchmark.py`는 설정마다 Pi가 55°C 아래로 식을 때까지 기다립니다(`--cool-to`). 스로틀링이 생긴 측정은 `valid=False`로 표시되고 최종 표에서 빠집니다.
- 메모리가 부족하면 `04_benchmark.py --frames 80`.
- NCNN 변환은 처음 한 번 인터넷이 필요합니다(`ncnn`, `pnnx` 설치).
- 녹화 영상으로 `07_live.py --device recordings/S4_walk_01.avi`를 돌리면 카메라 없이 동작을 미리 확인할 수 있습니다(이때 온도·읽기 시간은 의미 없음).
