from ultralytics import YOLO
import os
import onnxruntime as ort

PT_DIR = "./pytorch_models"
ONNX_DIR = "./onnx_models"

MODEL_NAMES = ["yolov8n", "yolo26n"]
RESOLUTIONS = [640, 960, 1088]

os.makedirs(PT_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

models = []

# 1. PyTorch 다운로드 및 ONNX 변환
for name in MODEL_NAMES:

    pt_path = os.path.join(PT_DIR, f"{name}.pt")

    # 모델이 없으면 자동 다운로드
    model = YOLO(pt_path)

    for res in RESOLUTIONS:

        onnx_path = os.path.join(
            ONNX_DIR, f"{name}_{res}.onnx"
        )

        # 이미 변환된 파일이 없을 때만 변환
        if not os.path.exists(onnx_path):

            exported_path = model.export(
                format="onnx",
                imgsz=res,
                batch=1,
                dynamic=False,
                simplify=True,
                opset=17,
                device="cpu"
            )

            os.replace(exported_path, onnx_path)

        models.append(onnx_path)

    del model

# 2. ONNX 모델 정보 확인
for path in models:

    size = os.path.getsize(path) / 1e6

    sess = ort.InferenceSession(
        path,
        providers=["CPUExecutionProvider"]
    )

    print("-" * 50)
    print("Model name:", os.path.basename(path))
    print(f"Model size: {size:.2f} MB")
    print("Input shape:", sess.get_inputs()[0].shape)
    print("Input type:", sess.get_inputs()[0].type)
    print("Output shape:", sess.get_outputs()[0].shape)

    del sess