import cv2
import os
from pathlib import Path
from PIL import Image

def generate_optimized_gif(
    video_path: str,
    output_gif_path: str,
    target_width: int = 900,
    fps_out: int = 8,
):
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps
    print(f"Source: {video.name} | {total_frames} frames | {duration:.2f}s @ {source_fps} fps")

    # Dynamic time-segments: (start_sec, end_sec, target_screen_duration_in_gif)
    # Optimized for brisk portfolio showcase: ~24s total playback, ~190 frames, <8MB file size
    segments = [
        (0.0, 12.0, 3.0),   # 0s-12s: Uploading & selecting file -> 3.0s
        (12.0, 18.0, 3.0),  # 12s-18s: Step 1: 3-panel OpenCV rectification -> 3.0s
        (18.0, 27.0, 3.5),  # 18s-27s: Step 2: Extracted parameter matrix table -> 3.5s
        (27.0, 39.0, 3.5),  # 27s-39s: Step 3 & 4: Triage ACCEPTED & Neo4j Ingestion -> 3.5s
        (39.0, 49.0, 2.5),  # 39s-49s: Tab 2 switch & prompt selection -> 2.5s
        (49.0, 56.0, 1.5),  # 49s-56s: Graph-RAG spinner running -> 1.5s
        (56.0, 68.0, 4.0),  # 56s-68s: Grounded Gemini Answer -> 4.0s
        (68.0, 78.5, 3.5),  # 68s-78.5s: Asset Inventory table & metric cards -> 3.5s
    ]

    selected_frame_indices = []
    
    # Calculate sample indices based on segment target durations and output fps
    for start_t, end_t, target_dur in segments:
        num_output_frames = max(1, int(target_dur * fps_out))
        start_frame = int(start_t * source_fps)
        end_frame = min(int(end_t * source_fps), total_frames - 1)
        
        step = (end_frame - start_frame) / num_output_frames
        for i in range(num_output_frames):
            idx = int(start_frame + i * step)
            if idx < total_frames and (not selected_frame_indices or idx > selected_frame_indices[-1]):
                selected_frame_indices.append(idx)

    print(f"Total sampled frames for GIF: {len(selected_frame_indices)}")
    
    frames_pil = []
    for count, f_idx in enumerate(selected_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        
        # Calculate proportional height
        h, w = frame_bgr.shape[:2]
        target_height = int(h * (target_width / w))
        
        # High quality resizing
        resized_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        img_pil = Image.fromarray(frame_rgb)
        
        # Quantize to adaptive 256 colors with dithering for smooth dark-mode gradients
        img_quantized = img_pil.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        frames_pil.append(img_quantized)
        
        if (count + 1) % 40 == 0:
            print(f"  Processed {count + 1}/{len(selected_frame_indices)} frames...")

    cap.release()

    if not frames_pil:
        raise RuntimeError("No frames extracted from video.")

    print(f"Saving optimized GIF to {output_gif_path}...")
    output_path = Path(output_gif_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Frame duration in milliseconds (1000ms / fps)
    frame_duration = int(1000 / fps_out)

    # Save animated GIF
    frames_pil[0].save(
        str(output_path),
        save_all=True,
        append_images=frames_pil[1:],
        optimize=True,
        duration=frame_duration,
        loop=0,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] GIF generated successfully: {output_path} ({size_mb:.2f} MB, {len(frames_pil)} frames, ~{len(frames_pil)/fps_out:.1f}s)")
    return size_mb

if __name__ == "__main__":
    video = r"C:\Users\shivam kumar jha\Downloads\Recording 2026-10-18 223615.mp4"
    out = r"snapshots/demo_walkthrough.gif"
    generate_optimized_gif(video, out, target_width=900, fps_out=8)
