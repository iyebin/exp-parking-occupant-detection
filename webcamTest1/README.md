# ExitGuard 카메라 사전 테스트 — Razer Kiyo Pro / Camera Module 3 NoIR + Raspberry Pi 4

YOLOv8n과 YOLO26n이 Pi 4 CPU에서 사람을 얼마나 빠르고 정확하게, 얼마나 멀리서 찾는지 측정합니다.
측정 항목: 모델 크기, 추론 속도, 단계별 병목, 정확도·인원 카운트, 온도, 검출 거리, 지연.

**원칙: 한 번 녹화하고, 모든 설정에 같은 영상을 다시 돌립니다.** 정확도·거리·속도·병목은 녹화 영상으로(오프라인),
지연과 발열만 카메라를 켜고(라이브) 잽니다. 그래야 모델끼리 공정하게 비교됩니다.

지원하는 카메라는 두 가지입니다. 같은 코드로 둘 다 돌아가며, 환경 변수 하나로 고릅니다(2-1절).

| 카메라 | 연결 | 선택 방법 |
|---|---|---|
| Razer Kiyo Pro (RGB, USB 웹캠) | 파란색 USB 3.0 포트 | 기본값 (아무것도 안 해도 됨) |
| Raspberry Pi Camera Module 3 Wide **NoIR** (`imx708_wide_noir`) | 리본 케이블(CSI) | `export EXITGUARD_CAM=picam` |

> **바로 라이브로 사람 검출을 확인하려면 → [5절 라이브 검출 확인 (`live_check.py`)](#5-라이브-검출-확인-live_checkpy)**

---

## 1. 준비물

| 장비 | 용도 | 비고 |
|---|---|---|
| Raspberry Pi 4B (RAM 4 GB 이상 권장) | 측정 대상 | Raspberry Pi OS **64-bit** |
| 공식 5V 3A USB-C 전원 | 전원 | 약한 어댑터는 저전압 → 속도 저하 |
| 팬(+방열판) | 냉각 | 팬 있음/없음, 케이스 열림/닫힘을 기록 |
| Razer Kiyo Pro (RZ19-0364) **또는** Camera Module 3 NoIR | 카메라 | Kiyo: **파란색 USB 3.0 포트** / NoIR: CSI 리본 케이블 |
| (NoIR, 어두운 환경) 적외선(IR) 조명 | 야간 촬영 | NoIR은 IR 조명이 있어야 어두운 곳에서 보입니다 |
| 삼각대, 줄자, 바닥 테이프 | 거리 표시 | 0.5~8 m 표시 |
| 3색 LED 모듈(KS0310) | 지연 측정(선택) | 빨강 핀 → GPIO17 |
| 노트북 | SSH, 라벨링, 결과 보기 | 아래 3절 |

## 2. Pi 환경 설정 (처음 한 번)

```bash
# 저장소를 Pi에 clone 한 뒤, 이 README가 있는 폴더로 이동
cd ~/exp-parking-occupant-detection/webcamTest1
bash setup_pi.sh            # apt 패키지, 가상환경(~/exitguard-venv), pip 패키지, 카메라 확인
sudo reboot                 # video 그룹 적용
```

가상환경은 프로젝트 폴더 안(`.venv`)이 아니라 **홈 폴더의 `~/exitguard-venv`** 에 만들어집니다.
VS Code에서는 `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path…** → `/home/<사용자>/exitguard-venv/bin/python`.

테스트할 때마다 (새 터미널을 열 때마다):

```bash
source ~/exitguard-venv/bin/activate            # 프롬프트 앞에 (exitguard-venv) 가 보이면 OK
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   # CPU 최고 속도 고정 (재부팅하면 풀림)
vcgencmd get_throttled      # throttled=0x0 이어야 정상 (저전압·과열 기록 없음)
python check_env.py         # 전부 OK 이고 마지막 줄이 READY 이면 준비 끝
```

### 2-1. NoIR 카메라(Camera Module 3 NoIR)를 쓸 때

Kiyo Pro 대신 CSI 카메라를 쓰는 Pi에서는 아래 두 줄을 **매 터미널마다** 실행합니다.

```bash
export EXITGUARD_CAM=picam              # 모든 스크립트가 Picamera2로 NoIR 카메라를 엶
export LIBCAMERA_LOG_LEVELS=*:ERROR     # libcamera INFO/WARN 로그 숨김 (선택)
```

매번 치기 번거로우면 한 번만 등록해 두세요 (NoIR Pi에서만):

```bash
echo 'export EXITGUARD_CAM=picam' >> ~/.bashrc
echo 'export LIBCAMERA_LOG_LEVELS=*:ERROR' >> ~/.bashrc
```

확인 순서:

```bash
rpicam-hello --list-cameras             # "imx708_wide_noir" 가 보여야 함
python3 -c "import picamera2; print('picamera2 OK')"   # 실패하면: sudo apt install -y python3-picamera2
python check_env.py                     # == camera (Pi CSI camera) 아래가 모두 OK
python 00_camera_check.py               # 1280x720@30 에서 약 30 fps, failed reads 0
```

NoIR 주의 사항
- **카메라는 한 번에 한 프로그램만** 쓸 수 있습니다. `Device or resource busy` / `Pipeline handler in use by another process`
  가 나오면 다른 스크립트나 `rpicam-*` 를 끄세요: `ps aux | grep -E "rpicam|python" | grep -v grep` → `kill <PID>` (안 되면 재부팅).
- SSH로 접속해 화면이 없으므로 `rpicam-still`/`rpicam-hello` 에는 `-n`(미리보기 끔)을 붙이세요: `rpicam-still -n -o test.jpg`
- 낮에는 화면이 **분홍/보라색**으로 보이는 것이 정상입니다(IR 필터 없음). 정확도에 영향이 있을 수 있으니 결과에 기록하세요.
- NoIR 튜닝 파일(`imx708_wide_noir.json`)은 자동으로 선택됩니다. 자동 초점은 연속(Continuous) 모드입니다.
- 카메라가 색 변환을 하드웨어(ISP)에서 하므로 `colour` 시간은 거의 0 ms 로 나옵니다. Kiyo 결과와 비교할 때 참고하세요.
- 1920x1080@60 은 Pi 4에서 약 36 fps 까지만 나옵니다. 1280x720@30 또는 1920x1080@30 을 쓰세요.
- `00_camera_check.py` 가 저장하는 `v4l2_*.txt` 는 USB 웹캠용이라 NoIR 에서는 의미가 없습니다.

**카메라 주의 (Kiyo Pro + Linux)**
- 리눅스 커널에 이 카메라 전용 보정(USB 절전 끄기, NO_LPM)이 들어가 있습니다. 카메라가 자꾸 끊기면
  `/boot/firmware/cmdline.txt` 끝에 `usbcore.quirks=1532:0e05:k` 를 추가하고 재부팅하세요.
- `v4l2-ctl`로 설정(특히 초점)을 바꾸다 카메라가 멈췄다는 보고가 있습니다. **한 번에 하나씩** 바꾸고, 멈추면 뽑았다 다시 꽂으세요.
- 화각(103°/90°/80°)은 Windows의 Razer Synapse에서 바꾸는 기능입니다. 리눅스에서 바뀌는지는
  `results/camera/v4l2_controls.txt`에서 zoom 계열 항목을 확인하세요. 안 되면 기본 화각을 줄자로 확인해 기록합니다.

## 3. 노트북 환경

| 할 일 | 필요한 것 |
|---|---|
| Pi 원격 접속 | SSH (VS Code Remote-SSH 추천 — 라이브 뷰어 포트도 자동으로 연결됨) |
| 라이브 영상 보기 | 웹 브라우저 (`live_check.py` 실행 후 http://localhost:8080) |
| 라벨링 | CVAT(app.cvat.ai), Roboflow, Label Studio 중 하나 — 클래스 `person` 하나, YOLO 형식으로 내보내기 |
| 결과 보기 | 엑셀/구글 시트, 또는 `python 09_summarize.py` |
| (선택) 1080p 녹화를 노트북에서 | Python 3.10+, `pip install -r requirements.txt`, `python 02_record.py --device 0 ...` |
| (선택) 카메라 펌웨어·화각 | Windows + Razer Synapse |
| (선택) 스크립트 미리 돌려보기 | 영상 파일로 실행 가능: `python 07_live.py --device clip.avi ...` |

## 4. 폴더 구조

```text
webcamTest1/
  common.py              공통 함수(모델 경로, 카메라 열기[Kiyo/NoIR], 온도 읽기, CSV)
  check_env.py           환경 점검 (EXITGUARD_CAM=picam 이면 NoIR 카메라를 점검)
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
  live_check.py          라이브 사람 검출 빠른 확인 + 브라우저 뷰어 (5절)
  models/  recordings/  labeled/  results/      (실행하면 생김, git 에는 올리지 않음)
```

## 5. 라이브 검출 확인 (`live_check.py`)

본 측정(6절)에 들어가기 전에, **카메라를 켜고 YOLO가 실제로 사람을 찾는지** 눈으로 확인하는 단계입니다.
모델마다 정해진 시간 동안 돌리면서 터미널에 매초 FPS와 검출된 사람 수를 찍고,
박스가 그려진 영상을 브라우저로 보여 주고, 가장 잘 잡힌 장면을 이미지로 저장합니다.

### 5-1. 이번 테스트에서 쓰는 설정 (이 4개만 사용)

| 설정 이름 | 모델 | 형식 | 입력 크기 |
|---|---|---|---|
| `yolo26n_onnx_640` | YOLO26n | ONNX | 640 |
| `yolo26n_onnx_320` | YOLO26n | ONNX | 320 |
| `yolov8n_onnx_640` | YOLOv8n | ONNX | 640 |
| `yolov8n_onnx_320` | YOLOv8n | ONNX | 320 |

- **모델은 `yolo26n`, `yolov8n` 두 개, 형식은 `onnx` 만, 입력 크기는 `320`, `640` 두 개만** 씁니다.
  PyTorch(`pt`), NCNN, `_e2e`, 416/960/1088 은 이 단계에서 쓰지 않습니다.
- `live_check.py` 의 기본값이 이미 이 4개입니다. 옵션 없이 실행하면 됩니다.
- 입력 크기 640 은 작은(먼) 사람을 더 잘 찾지만 느리고, 320 은 빠르지만 먼 사람을 놓치기 쉽습니다. 둘을 비교하는 것이 목적입니다.

### 5-2. 준비 (처음 한 번)

1) 가상환경 켜기, (NoIR이면) 카메라 선택

```bash
cd ~/exp-parking-occupant-detection/webcamTest1
source ~/exitguard-venv/bin/activate
export EXITGUARD_CAM=picam              # NoIR 카메라일 때만. Kiyo Pro면 이 줄 생략
export LIBCAMERA_LOG_LEVELS=*:ERROR     # 선택
```

2) ONNX 모델 만들기 — 필요한 4개만 변환

```bash
python 01_export_models.py --models yolo26n yolov8n --formats onnx --sizes 320 640
```

3) 모델 파일이 있는지 확인 — 아래 4개가 보여야 합니다

```bash
ls models/*.onnx
# models/yolo26n_320.onnx  models/yolo26n_640.onnx  models/yolov8n_320.onnx  models/yolov8n_640.onnx
```

없는 파일이 있으면 `live_check.py` 가 그 설정을 `skipped` 로 건너뜁니다.

4) 카메라가 프레임을 주는지 확인

```bash
python check_env.py       # camera gives frames ... 1280x720 이 OK 여야 함
```

### 5-3. 실행

```bash
python live_check.py
```

기본 동작: `yolo26n_640 → yolo26n_320 → yolov8n_640 → yolov8n_320` 순서로 **각 20초**, 카메라 1280x720@30, 신뢰도 기준 0.4.
카메라는 한 번만 열고 4개 설정이 같은 카메라를 이어서 씁니다. 전체 약 1분 30초 걸립니다.

**실행하는 동안 카메라 앞(약 1~5 m)에 사람이 서 있거나 걸어 다녀야** 검출 여부를 확인할 수 있습니다.
빈 화면에서도 한 번 돌려 보면 오검출(사람이 없는데 있다고 함)도 확인할 수 있습니다.

중간에 멈추려면 **Ctrl+C**. 멈춰도 그때까지의 요약표는 출력됩니다.

### 5-4. 브라우저 뷰어로 보기

실행하면 터미널에 다음 줄이 나옵니다.

```text
viewer: http://localhost:8080  (VS Code PORTS tab, or http://<pi-ip>:8080)
```

- **VS Code Remote-SSH 로 접속한 경우:** 노트북 브라우저에서 **http://localhost:8080** 을 엽니다.
  VS Code가 포트를 자동으로 노트북으로 연결합니다. 안 열리면 VS Code 아래쪽 패널의 **PORTS** 탭 → **Forward a Port** → `8080` 입력 후 다시 엽니다.
- **같은 네트워크의 다른 기기에서:** `http://<Pi의 IP>:8080` (Pi IP 확인: `hostname -I`)
- 화면 내용: 사람마다 **초록 박스 + `person 0.87`(신뢰도)**, 왼쪽 위에 **현재 설정 이름과 검출 인원 수**(빨간 글씨).
  설정이 바뀌면 왼쪽 위 이름도 바뀝니다.
- 뷰어는 스크립트가 실행되는 동안만 열려 있습니다.

### 5-5. 터미널 출력 읽는 법

매초 한 줄씩 출력됩니다. (아래 숫자는 예시이며 실제 값은 다를 수 있습니다)

```text
[yolo26n_onnx_640] loading ...
    1.0s  fps   3.1  infer  301.2 ms  persons 1  conf 0.86
    2.0s  fps   3.2  infer  298.7 ms  persons 2  conf 0.91 0.64
```

| 항목 | 의미 |
|---|---|
| `fps` | 지난 1초 동안 처리한 프레임 수 (카메라 읽기 + 추론 + 박스 그리기 포함) |
| `infer` | 마지막 프레임의 추론 시간(ms) |
| `persons` | 마지막 프레임에서 찾은 사람 수 (신뢰도 0.4 이상) |
| `conf` | 각 사람의 신뢰도 (1.0에 가까울수록 확실) |

**사람이 앞에 있는데 `persons 0` 이 계속 나오면** YOLO가 못 찾고 있는 것입니다 → 5-8절 문제 해결.

4개 설정이 끝나면 요약표가 나옵니다.

```text
== summary  (FPS includes JPEG encoding for the viewer; use --no-stream for clean numbers)
config                    fps  infer ms  frames w/ person  best frame
yolo26n_onnx_640          3.1     300.5     58/62    ( 94%)  results/live_check/yolo26n_onnx_640_best.jpg
yolo26n_onnx_320          ...
```

| 열 | 의미 |
|---|---|
| `fps` | 20초 평균 처리 속도 |
| `infer ms` | 추론 시간 중앙값 |
| `frames w/ person` | 사람이 1명 이상 검출된 프레임 수 / 전체 프레임 수 (%). 계속 사람이 서 있었다면 100%에 가까워야 좋음 |
| `best frame` | 신뢰도가 가장 높았던 장면 이미지 |

### 5-6. 저장되는 결과

- `results/live_check/<설정>_best.jpg` — 설정마다 가장 잘 검출된 장면(박스 포함). VS Code 탐색기에서 클릭해서 볼 수 있습니다.
- 요약표는 터미널에만 출력됩니다. 기록이 필요하면 파일로도 남기세요:

```bash
python live_check.py --no-stream 2>&1 | tee results/live_check/summary_$(date +%Y%m%d_%H%M).txt
```

### 5-7. 권장 순서와 옵션

1. **눈으로 확인:** `python live_check.py` → 브라우저에서 박스가 사람을 따라다니는지 확인
2. **보고용 숫자:** `python live_check.py --no-stream --seconds 60`
   → 뷰어용 JPEG 압축을 하지 않으므로 FPS가 더 정확합니다. 표에 적는 숫자는 이것을 쓰세요.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models` | `yolo26n yolov8n` | 돌릴 모델. 이번 테스트는 이 두 개만 |
| `--fmt` | `onnx` | 모델 형식. 이번 테스트는 `onnx` 만 |
| `--sizes` | `640 320` | 입력 크기. 이번 테스트는 `320`, `640` 만 |
| `--seconds` | `20` | 설정 하나당 실행 시간(초) |
| `--conf` | `0.4` | 이 신뢰도 이상만 사람으로 셈 |
| `--res` | `1280x720` | 카메라 해상도 |
| `--device` | `/dev/video0` | Kiyo 장치, 영상 파일, 또는 `picam` (`EXITGUARD_CAM=picam` 대신 사용 가능) |
| `--port` | `8080` | 뷰어 포트 |
| `--no-stream` | 꺼짐 | 뷰어 끔 (FPS 측정용) |

예시:

```bash
python live_check.py --sizes 320                       # 320 만
python live_check.py --models yolo26n --sizes 640      # yolo26n 640 하나만
python live_check.py --seconds 60 --no-stream          # 보고용 측정
python live_check.py --device picam                    # export 없이 NoIR 사용
python live_check.py --device recordings/S4_walk_01.avi  # 카메라 없이 녹화 영상으로
```

### 5-8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `cannot open Pi camera ... busy` | 다른 프로그램이 카메라 사용 중. `ps aux \| grep -E "rpicam\|python" \| grep -v grep` → `kill <PID>`, 안 되면 재부팅 |
| `another test script is running` | `07_live.py` 등 다른 측정 스크립트가 돌고 있음. 끄고 다시 실행 |
| `[...] skipped - ... missing` | 그 ONNX 모델이 없음 → 5-2의 `01_export_models.py` 실행 |
| `cannot open camera /dev/video0` (NoIR Pi) | `export EXITGUARD_CAM=picam` 을 안 함 |
| 브라우저에서 안 열림 | VS Code **PORTS** 탭에서 8080 추가, 또는 `http://<Pi IP>:8080`. 스크립트가 실행 중인지 확인 |
| 사람이 있는데 `persons 0` | 조명이 너무 어두움(NoIR은 IR 조명 필요), 너무 멀거나 화면 가장자리, 입력 크기 320 → 640 으로 비교, `--conf 0.25` 로 낮춰 보기 |
| FPS가 너무 낮음 | CPU governor `performance` 설정, `vcgencmd get_throttled` 확인(과열·저전압), `--no-stream` 사용 |
| 화면이 분홍/보라색 | NoIR 카메라의 정상 동작(IR 필터 없음) |

## 6. 본 측정 실행 순서

NoIR Pi에서는 모든 단계 전에 `export EXITGUARD_CAM=picam` 을 해 두세요. 같은 명령이 그대로 NoIR 카메라로 동작합니다.

| 순서 | 명령 | 결과 파일 | 측정 항목 |
|---|---|---|---|
| 0 | `python check_env.py` | — | 환경 점검 (READY 확인) |
| 1 | `python 00_camera_check.py` | `results/camera/` | 카메라 단독 FPS, 읽기·색 변환 시간, 끊김 |
| 1+ | `python 00_camera_check.py --modes 1280x720@30 --duration 600` | 〃 | 10분 연속 안정성 (Kiyo: + `dmesg \| grep -i uvc`) |
| 2 | `python 01_export_models.py` | `models/`, `results/models.csv` | **모델 크기(MB)**, 파라미터, GFLOPs |
| 2+ | `python live_check.py` | `results/live_check/` | **라이브 검출 빠른 확인** (5절) |
| 3 | `python 02_record.py --name ... --seconds ...` (시나리오마다) | `recordings/*.avi` | — |
| 4 | `python 04_benchmark.py` | `results/bench.csv` | **추론 속도(ms)**, **병목(ms)**, CPU·메모리 |
| 5 | `python 03_extract_frames.py --per-clip 20` → 라벨링 → `python 05_accuracy.py` | `results/accuracy*.csv` | **정확도**, 인원 카운트, 빈 화면 오탐 |
| 6 | `python 06_distance.py` | `results/distance*.csv` | **검출 거리** |
| 7 | `python 07_live.py ... --duration 60` | `results/live/` | **지연**, 카메라 포함 전체 병목 |
| 8 | `python 07_live.py ... --duration 900 --cooling fan --case open --ambient 22` | `results/live/` | **온도**(15분), 스로틀링, FPS 변화 |
| 9 | (선택) `python 08_led_latency.py --pick-roi` → `--roi ...` | `results/led_latency.csv` | 실제 카메라 지연 |
| 10 | `python 09_summarize.py` | `results/final_table.*`, 온도 그래프 | 전체 표 |

### 비교하는 설정

이번 테스트는 5절과 같이 **YOLO26n·YOLOv8n × ONNX × 320/640 (4개)** 로 진행합니다.
04~08 스크립트에도 같은 범위를 지정하세요:

```bash
python 04_benchmark.py --models yolo26n yolov8n --formats onnx --sizes 320 640
```

전체 비교(참고용, 총 24개):

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
(NoIR은 1920x1080 에서 30 fps 까지 안정적입니다.)

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
(NoIR 카메라는 노트북에 연결할 수 없으므로 Pi에서 1280x720 으로 녹화하세요.)

### 라벨링

1. `python 03_extract_frames.py --per-clip 20` → `labeled/images/`
2. CVAT 등에서 `person` 한 클래스로 박스를 그림. **사람이 없는 이미지도 빈 라벨로** 내보내야 오탐을 셀 수 있습니다.
3. YOLO 형식 txt를 `labeled/labels/`에 (이미지와 같은 파일 이름)

## 7. 결과 읽는 법

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

## 8. 알아둘 점

- `04~06`은 `--source-res 1280x720`으로 라이브 캡처 크기에 맞춰 줄인 프레임을 씁니다. 1080p로 돌리려면 `--source-res none`.
- `04_benchmark.py`는 설정마다 Pi가 55°C 아래로 식을 때까지 기다립니다(`--cool-to`). 스로틀링이 생긴 측정은 `valid=False`로 표시되고 최종 표에서 빠집니다.
- 메모리가 부족하면 `04_benchmark.py --frames 80`.
- NCNN 변환은 처음 한 번 인터넷이 필요합니다(`ncnn`, `pnnx` 설치).
- 녹화 영상으로 `07_live.py --device recordings/S4_walk_01.avi`를 돌리면 카메라 없이 동작을 미리 확인할 수 있습니다(이때 온도·읽기 시간은 의미 없음).
- 카메라를 쓰는 스크립트(`00`, `02`, `07`, `08`, `live_check.py`, `preview.py`)는 **동시에 하나만** 실행하세요.
- `models/`, `recordings/`, `results/`, `*.log` 는 `.gitignore` 로 git 에 올리지 않습니다. 새 Pi에서는 `01_export_models.py` 로 모델을 다시 만드세요.