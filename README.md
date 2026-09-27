# Real-Time ROI Frame Capture with YOLOv8

> **Take-home test assignment for a job interview** (computer vision). The final solution is at the top level; the development history is in [`drafts/`](drafts).

Real-time video analytics that **captures a snapshot of a region of interest (ROI) whenever a target object (a car) enters it**, e.g. to record vehicles crossing a checkpoint or to trigger a camera / motor stop.

## How it works

1. Video frames are read with OpenCV (a video file or a camera stream).
2. **YOLOv8s** detects objects; only the `car` class is used.
3. If a car's bounding box intersects the ROI (or lies fully inside it), the ROI is cropped and saved to `saved_frames/frame_<n>.jpg`.
4. The ROI and the detections are drawn on the live preview.

### Keeping it real-time

The batched version (`roi_capture_batched.ipynb`) keeps up with the stream on a regular machine by:

| Technique | Effect |
|---|---|
| Processing every 2nd frame | halves the detection load |
| Downscaling frames by 20% | faster inference |
| Batched inference (8 frames) | better GPU/CPU utilisation; larger batches slowed things down |
| Batches processed in a separate thread | reading the stream is not blocked by inference |

YOLOv8s was chosen as the trade-off: smaller models missed objects, while larger v8 models could not keep up in real time.

## Files

| File | Description |
|---|---|
| `roi_capture_batched.ipynb` | Optimised version: frame skipping, batching, multithreading |
| `roi_capture.py` | Simple sequential version |

## Quick start

```bash
pip install -r requirements.txt
python roi_capture.py
```

- Set the input in `cv2.VideoCapture(...)`: a path to a video file, or `0` for a webcam.
- Adjust the ROI with `roi = [(x1, y1), (x2, y2)]`.
- YOLOv8 weights are downloaded automatically on the first run. Press **`q`** to quit.

## Development process

This was a take-home task for a job interview. [`drafts/`](drafts) keeps the experiments that led to the batched, multithreaded version:

| Files | Idea |
|---|---|
| `4.ipynb`, `4 copy.ipynb`, `4-туц.ipynb`, `4-tini.ipynb` | First detection pipelines, including a YOLO *tiny* model for speed |
| `5.ipynb`, `8.ipynb`, `8л.ipynb`, `8х.ipynb`, `itog8с.ipynb` | Frame skipping, batching and threading variants |
| `ухудшенаяверсиясбатчамию.py` | A batched variant that turned out slower, kept as a negative result |

## Tech stack

Python · Ultralytics YOLOv8 · OpenCV · threading

## License

[MIT](LICENSE)
