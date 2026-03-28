import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import XCLIPProcessor, XCLIPModel
import cv2

JSON_FILE = "dataset_pickleball/pickleball_caption.json"
VIDEO_DIR = "dataset_pickleball/videos"
MODEL_NAME = "microsoft/xclip-base-patch32"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
NUM_FRAMES = 8
GRAD_ACCUM_STEPS = 4
SAVE_DIR = "pickleball_xclip_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

class PickleballXCLIPDataset(Dataset):
    def __init__(self, json_file, video_dir, num_frames=8):
        self.video_dir = video_dir
        self.num_frames = num_frames

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.records = data.get("sentences", [])

        self.valid_records = [
            item for item in self.records
            if os.path.exists(os.path.join(self.video_dir, f"{item['video_id']}.mp4"))
        ]
        print(f"Dataset có {len(self.valid_records)} videos hợp lệ.")

        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
        ).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.valid_records)

    def _load_video_opencv(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Không mở được video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video không có frame: {video_path}")

        indices = torch.linspace(0, total_frames - 1, self.num_frames).long().tolist()
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    cap.release()
                    raise ValueError(f"Không đọc được frame {idx} của {video_path}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()
        video_frames = torch.stack(frames, dim=0)  # (F, C, H, W)
        return video_frames

    def __getitem__(self, idx):
        item = self.valid_records[idx]
        video_path = os.path.join(self.video_dir, f"{item['video_id']}.mp4")
        caption = item["caption"]

        try:
            video_frames = self._load_video_opencv(video_path)
            video_frames = F.interpolate(
                video_frames,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )
            video_frames = (video_frames - self.mean) / self.std

        except Exception as e:
            print(f"Lỗi đọc video {video_path}: {e}")
            return None

        return {
            "video": video_frames,
            "text": caption,
            "video_id": item["video_id"]
        }

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    videos = torch.stack([item["video"] for item in batch], dim=0)  # (B, F, C, H, W)
    texts = [item["text"] for item in batch]
    video_ids = [item["video_id"] for item in batch]
    return videos, texts, video_ids

def freeze_backbone_except_projection(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

    keywords = [
        "visual_projection",
        "text_projection",
        "logit_scale",
        "prompts_generator",
        "mit"
    ]

    for name, param in model.named_parameters():
        if any(k in name for k in keywords):
            param.requires_grad = True

def train():
    print("Đang tải model X-CLIP từ Hugging Face...")
    processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
    model = XCLIPModel.from_pretrained(MODEL_NAME).to(device)

    freeze_backbone_except_projection(model)

    dataset = PickleballXCLIPDataset(
        JSON_FILE,
        VIDEO_DIR,
        num_frames=NUM_FRAMES
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Số lượng trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    print("Bắt đầu huấn luyện...")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_valid_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            if batch is None:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{step+1}/{len(dataloader)}] | Skip batch rỗng")
                continue

            videos, texts, video_ids = batch

            if len(set(video_ids)) < len(video_ids):
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{step+1}/{len(dataloader)}] | Skip duplicate video_id")
                continue

            text_inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )

            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_inputs["pixel_values"] = videos.to(device, non_blocking=True)

            outputs = model(**text_inputs)

            logits_per_video = outputs.logits_per_video
            logits_per_text = outputs.logits_per_text

            bs = logits_per_video.size(0)
            labels = torch.arange(bs, device=device)

            loss_v = loss_fn(logits_per_video, labels)
            loss_t = loss_fn(logits_per_text, labels)
            loss = (loss_v + loss_t) / 2

            scaled_loss = loss / GRAD_ACCUM_STEPS
            scaled_loss.backward()

            if (num_valid_steps + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            num_valid_steps += 1

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                f"Batch [{step+1}/{len(dataloader)}] | "
                f"Loss: {loss.item():.4f}"
            )

        if num_valid_steps > 0 and (num_valid_steps % GRAD_ACCUM_STEPS != 0):
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, num_valid_steps)
        print(
            f"Kết thúc Epoch {epoch+1} - "
            f"Average Loss: {avg_loss:.4f} | "
            f"Valid steps: {num_valid_steps}\n"
        )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print(f"Đã lưu model thành công vào: {SAVE_DIR}")


if __name__ == "__main__":
    train()