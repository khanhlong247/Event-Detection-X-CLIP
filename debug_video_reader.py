import os
import sys
import traceback
import importlib.util

VIDEO_DIR = "dataset_pickleball/videos"
MAX_VIDEOS_TO_TEST = 10


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_module(name):
    spec = importlib.util.find_spec(name)
    return spec is not None


def inspect_environment():
    print_header("1. KIEM TRA MOI TRUONG")

    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version}")

    for module_name in ["numpy", "cv2", "torch"]:
        print(f"Module '{module_name}' installed: {check_module(module_name)}")

    try:
        import numpy as np
        print(f"NumPy version    : {np.__version__}")
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        print("NumPy basic test : OK", arr.shape, arr.dtype)
    except Exception as e:
        print(f"NumPy import/test FAILED: {e}")
        traceback.print_exc()

    try:
        import cv2
        print(f"OpenCV version   : {cv2.__version__}")
    except Exception as e:
        print(f"OpenCV import FAILED: {e}")
        traceback.print_exc()

    try:
        import torch
        print(f"PyTorch version  : {torch.__version__}")
        print(f"CUDA available   : {torch.cuda.is_available()}")
    except Exception as e:
        print(f"PyTorch import FAILED: {e}")
        traceback.print_exc()


def test_numpy_torch_bridge():
    print_header("2. KIEM TRA CAU NOI NUMPY -> TORCH")

    try:
        import numpy as np
        import torch

        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        print("Created numpy array successfully.")

        tensor = torch.from_numpy(arr)
        print("torch.from_numpy SUCCESS")
        print("Tensor shape:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)

    except Exception as e:
        print(f"torch.from_numpy FAILED: {e}")
        traceback.print_exc()


def list_video_files(video_dir):
    if not os.path.isdir(video_dir):
        print(f"Khong tim thay thu muc video: {video_dir}")
        return []

    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    video_files.sort()
    return video_files


def test_opencv_video_read(video_path):
    print(f"\n--- Test video: {video_path}")

    try:
        import cv2
    except Exception as e:
        print(f"Import cv2 FAILED: {e}")
        return False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("VideoCapture: FAILED (khong mo duoc video)")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"VideoCapture: OK")
    print(f"Frame count : {total_frames}")
    print(f"Resolution  : {width}x{height}")
    print(f"FPS         : {fps}")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Doc frame dau tien: FAILED")
        return False

    print("Doc frame dau tien: OK")
    print(f"Frame type  : {type(frame)}")
    print(f"Frame shape : {frame.shape}")
    print(f"Frame dtype : {frame.dtype}")

    try:
        import torch
        tensor = torch.from_numpy(frame)
        print("torch.from_numpy(frame): OK")
        print("Tensor shape:", tensor.shape)
        print("Tensor dtype:", tensor.dtype)
    except Exception as e:
        print(f"torch.from_numpy(frame): FAILED -> {e}")
        traceback.print_exc()
        return False

    try:
        frame_list = frame.tolist()
        tensor2 = torch.tensor(frame_list, dtype=torch.uint8)
        print("Fallback torch.tensor(frame.tolist()): OK")
        print("Fallback tensor shape:", tensor2.shape)
    except Exception as e:
        print(f"Fallback torch.tensor(frame.tolist()) FAILED -> {e}")
        traceback.print_exc()
        return False

    return True


def main():
    inspect_environment()
    test_numpy_torch_bridge()

    print_header("3. KIEM TRA DOC VIDEO BANG OPENCV")

    video_files = list_video_files(VIDEO_DIR)
    print(f"Tim thay {len(video_files)} video trong: {VIDEO_DIR}")

    if len(video_files) == 0:
        return

    num_ok = 0
    num_fail = 0

    for video_path in video_files[:MAX_VIDEOS_TO_TEST]:
        ok = test_opencv_video_read(video_path)
        if ok:
            num_ok += 1
        else:
            num_fail += 1

    print_header("4. TONG KET")
    print(f"So video test thanh cong: {num_ok}")
    print(f"So video test loi       : {num_fail}")

    print("\nNeu cv2 doc duoc frame nhung torch.from_numpy bi loi 'Numpy is not available',")
    print("thi van de nam o moi truong cai dat NumPy / PyTorch, khong phai do file video.")


if __name__ == "__main__":
    main()