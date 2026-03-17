# Deepfake Detector (Images & Videos)

A complete, attractive deepfake detection app:
- FastAPI backend with image and video prediction endpoints
- PyTorch training pipeline (ResNet18) for your custom dataset
- React (Vite) frontend with a modern upload UI and confidence bars

The backend ships with a deterministic placeholder so you can test today. Swap in your trained model later.

## Prerequisites
- Python 3.10+
- Node 18+
- Windows PowerShell steps below (Linux/Mac similar)

## Backend: run API
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- GET `http://localhost:8000/health`
- POST `http://localhost:8000/predict/image` (multipart form-data: `file`)
- POST `http://localhost:8000/predict/video` (multipart form-data: `file`)

## Frontend: run UI
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

If your API runs elsewhere, set env in `frontend/.env`:
```
VITE_API_URL=http://localhost:8000
```

## Dataset layout (images)
```
training/
  data/
    train/
      real/ ...
      fake/ ...
    val/
      real/ ...
      fake/ ...
```
- Use JPG/PNG image files inside `real/` and `fake/` folders.
- You provide the dataset; ensure classes are named exactly `real` and `fake`.

## Prepare dataset (split + optional resize)
Place your raw images under `raw/real` and `raw/fake`, then run:
```powershell
. .venv\Scripts\Activate.ps1
python training/prepare_dataset.py --source raw --out training/data --val-ratio 0.2 --resize 224
```
This creates `training/data/train/{real,fake}` and `training/data/val/{real,fake}`.

## Generate a tiny synthetic dataset (quick try)
No data yet? Generate a small balanced set for testing:
```powershell
. .venv\Scripts\Activate.ps1
python training/generate_tiny_dataset.py --out training/data --num-per-class 50 --image-size 224 --val-ratio 0.2
```
Then train as usual.

## Generate tiny synthetic videos (optional)
Create a small video dataset to test the video endpoint:
```powershell
. .venv\Scripts\Activate.ps1
python training/generate_tiny_videos.py --out training/videos --num-per-class 8 --size 224 --frames 48 --fps 12 --val-ratio 0.25
```
The backend will handle videos by sampling frames when a real model is loaded; otherwise it uses the placeholder.

## Video (CNN+LSTM) training & evaluation (optional)
Prepare videos under:
```
training/videos/
  train/{real,fake}/*.mp4
  val/{real,fake}/*.mp4
```
Train video model:
```powershell
. .venv\Scripts\Activate.ps1
python -m training.video_train --data-dir training/videos --epochs 3 --batch-size 4 --frames 16 --image-size 224 --out-model models/deepfake_video_cnnlstm.pth
```
Evaluate:
```powershell
python -m training.video_eval --data-dir training/videos --model models/deepfake_video_cnnlstm.pth --frames 16 --image-size 224
```
The backend auto-loads `models/deepfake_video_cnnlstm.pth` if present and uses it for `/predict/video`. Images and webcam frames continue using the CNN image classifier.

## Train the model
```powershell
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training/train.py --data-dir training/data --epochs 10 --batch-size 32 --out-model models/deepfake_cnn.pth
```
Checkpoints will be saved to `models/deepfake_cnn.pth` (best val accuracy).

## Plug trained model into backend
Edit `backend/inference.py`:
1) Load the checkpoint in `__init__` (use `torch.load` and `build_model` with saved config)
2) Implement real preprocessing and softmax to produce `{ real: p1, fake: p2 }`

Until then, the placeholder returns stable pseudo-probabilities based on file content hashing.

## Notes
- CORS enabled for local development.
- For videos, a production model should sample frames with OpenCV and fuse per-frame logits.
- Consider test-time augmentation and balanced sampling if classes are imbalanced.

## License
For academic/demo purposes. Verify legality and consent for any dataset you use.
