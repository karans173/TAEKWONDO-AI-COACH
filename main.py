import os
import time
import tempfile
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import mediapipe as mp

from core.biomechanics import BiomechanicsEngine
from core.state_machine import StateMachine
from core.filters import OneEuroFilter, clean_sequence
from models.st_gcn import get_adjacency_matrix, TKD_GRU, TKD_LSTM, TKD_STGCN


MAX_FRAMES = 325

ONE_EURO_MIN_CUTOFF = 0.8
ONE_EURO_BETA = 0.08
ONE_EURO_D_CUTOFF = 1.0

MEDIAN_KERNEL = 5
MODEL_DIR = "weights"
TARGET_KICKS = ["frontkick", "roundhouse", "sidekick"]

FEATURE_COLUMNS = [
    "base_knee_angle", "base_pivot_angle", "striking_chamber_angle",
    "striking_foot_angle", "spine_angle", "striking_ankle_y",
    "base_ankle_y", "guard_dropped",
]

GRAPH_REORDER = [0, 1, 2, 3, 4, 7, 5, 6]

CLASS_MAPS = {
    "frontkick": {
        0: "Perfect Execution",
        1: "Error: Dropped Guard",
        2: "Error: Leaning Back",
        3: "Error: No Chamber",
        4: "Error: No Retraction",
    },
    "roundhouse": {
        0: "Perfect Execution",
        1: "Error: Dropped Guard",
        2: "Error: Lazy Chamber",
    },
    "sidekick": {
        0: "Perfect Execution",
        1: "Error: Dropped Guard",
        2: "Error: Foot Pointing Up",
        3: "Error: No Pivot",
        4: "Error: Off Balance",
    },
}


def download_mediapipe_model():
    """Download the same Heavy Pose Landmarker model used during dataset extraction."""
    model_dir = "pose_landmarker"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_heavy.task")

    if not os.path.exists(model_path):
        st.info("Downloading MediaPipe Pose Landmarker Heavy model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        )
        urllib.request.urlretrieve(url, model_path)

    return model_path


def build_model(model_name, num_classes):
    if model_name == "LSTM":
        return TKD_LSTM(num_classes)
    if model_name == "GRU":
        return TKD_GRU(num_classes)
    if model_name == "ST-GCN":
        A = get_adjacency_matrix()
        return TKD_STGCN(num_classes, A)
    raise ValueError(f"Unknown model name: {model_name}")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    preprocessing = {}

    for kick in TARGET_KICKS:
        candidate_files = []
        if os.path.exists(MODEL_DIR):
            for filename in os.listdir(MODEL_DIR):
                if filename.startswith(f"{kick}_") and filename.endswith(".pth"):
                    candidate_files.append(filename)

        if len(candidate_files) == 0:
            st.error(f"No checkpoint found for {kick} in '{MODEL_DIR}/'")
            continue

        if len(candidate_files) > 1:
            candidate_files.sort()
            st.warning(
                f"Multiple checkpoints found for {kick}: {candidate_files}. "
                f"Using the first one alphabetically."
            )

        weight_path = os.path.join(MODEL_DIR, candidate_files[0])
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        required_keys = [
            "model_name", "model_state_dict", "mean", "std",
            "max_frames", "feature_columns",
        ]
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            raise RuntimeError(f"Checkpoint {weight_path} is missing: {missing}")

        model_name = checkpoint["model_name"]
        model = build_model(model_name=model_name, num_classes=len(CLASS_MAPS[kick]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        models[kick] = model
        preprocessing[kick] = {
            "model_name": model_name,
            "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
            "std": np.asarray(checkpoint["std"], dtype=np.float32),
            "max_frames": int(checkpoint["max_frames"]),
            "feature_columns": list(checkpoint["feature_columns"]),
            "median_kernel": int(checkpoint.get("median_kernel", MEDIAN_KERNEL)),
            "one_euro": checkpoint.get("one_euro", {
                "min_cutoff": ONE_EURO_MIN_CUTOFF,
                "beta": ONE_EURO_BETA,
                "d_cutoff": ONE_EURO_D_CUTOFF,
            }),
            "graph_adjacency": checkpoint.get("graph_adjacency"),
        }
        print(f"Loaded {kick}: {model_name}")

    return models, preprocessing, device


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14),
    (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
]

def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(image, p1, p2, (0, 255, 0), 2)

    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

def prepare_tensor(cleaned_df, mean, std, feature_columns, max_frames):
    """Exactly match the training data representation."""
    missing_columns = [col for col in feature_columns if col not in cleaned_df.columns]
    if missing_columns:
        raise ValueError(f"Missing features in captured sequence: {missing_columns}")

    values = cleaned_df[feature_columns].to_numpy(dtype=np.float32)
    current_len = values.shape[0]
    
    if current_len == 0:
        raise ValueError("Captured sequence is empty.")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if len(mean) != values.shape[1] or len(std) != values.shape[1]:
        raise ValueError(f"Normalization params dimension mismatch. Sequence has {values.shape[1]}.")

    values = (values - mean) / np.maximum(std, 1e-8)

    values = values[:, GRAPH_REORDER]

    if current_len > max_frames:
        values = values[:max_frames]
        current_len = max_frames
    elif current_len < max_frames:
        padding_length = max_frames - current_len
        padding = np.repeat(values[-1:], padding_length, axis=0)
        values = np.vstack([values, padding])

    # Transform: [T, 8] -> [T, 4, 2] -> [2, T, 4]
    values = values.reshape(max_frames, 4, 2).transpose(2, 0, 1)
    
    x = torch.from_numpy(values).float().unsqueeze(0)
    lengths = torch.tensor([current_len], dtype=torch.long)
    return x, lengths


def run_analysis(video_source, target_kick, is_right_leg, models, preprocessing, device, display_placeholder):
    if target_kick not in models:
        st.error(f"No trained model loaded for {target_kick}.")
        return

    cap = cv2.VideoCapture(video_source)
    is_live = isinstance(video_source, int)

    if not cap.isOpened():
        st.error("Could not open video source.")
        return

    biomechanics = BiomechanicsEngine()
    state_machine = StateMachine()
    model_cfg = preprocessing[target_kick]
    one_euro_cfg = model_cfg["one_euro"]
    
    euro_filter = OneEuroFilter(
        min_cutoff=float(one_euro_cfg["min_cutoff"]),
        beta=float(one_euro_cfg["beta"]),
        d_cutoff=float(one_euro_cfg["d_cutoff"]),
    )

    final_captured_sequence = []
    countdown_start = time.monotonic()

    # MediaPipe setup
    model_path = download_mediapipe_model()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    # Video timing
    original_fps = cap.get(cv2.CAP_PROP_FPS) if not is_live else 30.0
    if original_fps == 0 or np.isnan(original_fps) or original_fps < 1:
        original_fps = 30.0

    frame_count = 0
    display_interval = 1.0 / 30.0
    last_display_time = 0.0

    foot_indices = {27, 28, 29, 30, 31, 32}
    last_valid_coords = np.zeros((33, 4), dtype=np.float32)

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                current_time = frame_count / original_fps
                timestamp_ms = int(current_time * 1000)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]
                    draw_custom_landmarks(frame_rgb, landmarks)

                    # Live-camera countdown
                    elapsed_real = time.monotonic() - countdown_start
                    if is_live and elapsed_real < 5.0:
                        seconds_left = int(5.0 - elapsed_real) + 1
                        cv2.putText(
                            frame_rgb, f"STEP BACK: {seconds_left}s",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 0), 4
                        )
                    else:
                        # Convert MediaPipe coordinates to pixel space
                        raw_coords = np.zeros((33, 4), dtype=np.float32)
                        for i, lm in enumerate(landmarks):
                            visibility = float(getattr(lm, "visibility", 1.0))
                            current_val = (lm.x * w, lm.y * h, lm.z * w, visibility)

                            if i in foot_indices:
                                out_of_bounds = (lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0)
                                hidden = (visibility < 0.3)
                                if (out_of_bounds or hidden) and frame_count > 1:
                                    raw_coords[i] = last_valid_coords[i]
                                else:
                                    raw_coords[i] = current_val
                                    last_valid_coords[i] = current_val
                            else:
                                raw_coords[i] = current_val
                                if visibility >= 0.3:
                                    last_valid_coords[i] = current_val

                        filtered_xyz = euro_filter(
                            raw_coords[:, :3].ravel(), current_time
                        ).reshape(-1, 3)

                        filtered_landmarks = np.column_stack([filtered_xyz, raw_coords[:, 3]])
                        ankle_idx = 28 if is_right_leg else 27

                        ankle_visibility = filtered_landmarks[ankle_idx, 3]
                        if ankle_visibility < 0.4:
                            cv2.putText(
                                frame_rgb, "LOW ANKLE VISIBILITY", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4
                            )

                        features_dict = biomechanics.extract_kinematic_chains(
                            landmarks=filtered_landmarks,
                            current_time=current_time,
                            is_right_leg=is_right_leg,
                        )

                        current_ankle = filtered_landmarks[ankle_idx, :3]
                        _, _, captured_seq = state_machine.update(
                            features_dict=features_dict,
                            current_ankle_pos=current_ankle,
                            current_time=current_time,
                            frame_height=h,
                        )

                        if captured_seq is not None:
                            final_captured_sequence = captured_seq
                            break

                render_time = time.monotonic()
                if render_time - last_display_time >= display_interval:
                    display_placeholder.image(frame_rgb, channels="RGB")
                    last_display_time = render_time

    finally:
        cap.release()

    if not final_captured_sequence and state_machine.current_state.name == "KICKING":
        captured_seq = list(state_machine.kick_buffer)
        final_peak = state_machine.peak_elevation
        kick_threshold = state_machine.KICK_HEIGHT_THRESHOLD * h

        if final_peak >= kick_threshold and len(captured_seq) >= 15:
            final_captured_sequence = captured_seq

    if len(final_captured_sequence) < 10:
        st.error("Kick sequence too short, invalid, or no complete kick detected.")
        return

    df_raw = pd.DataFrame(final_captured_sequence)
    try:
        df_clean = clean_sequence(df_raw, median_kernel=model_cfg["median_kernel"])
    except TypeError:
        df_clean = clean_sequence(df_raw)

    try:
        x, lengths = prepare_tensor(
            cleaned_df=df_clean,
            mean=model_cfg["mean"],
            std=model_cfg["std"],
            feature_columns=model_cfg["feature_columns"],
            max_frames=model_cfg["max_frames"],
        )
    except Exception as exc:
        st.error(f"Failed to prepare model input: {exc}")
        return

    x = x.to(device)
    lengths = lengths.to(device)

    model = models[target_kick]
    with torch.no_grad():
        outputs = model(x, lengths)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    result_text = CLASS_MAPS[target_kick][predicted_idx]

    model_name = model_cfg.get("model_name", "Loaded model")
    st.write(f"Model: **{model_name}**")

    if "Error" in result_text:
        st.error(f"Prediction: **{result_text}** (Confidence: {confidence * 100:.1f}%)")
    else:
        st.balloons()
        st.success(f"Prediction: **{result_text}** (Confidence: {confidence * 100:.1f}%)")

    probability_rows = [
        {
            "Class": class_name,
            "Probability (%)": float(probabilities[0, class_idx].item() * 100),
        }
        for class_idx, class_name in CLASS_MAPS[target_kick].items()
    ]

    probability_df = pd.DataFrame(probability_rows).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
    st.dataframe(probability_df, use_container_width=True, hide_index=True)


st.set_page_config(page_title="TKD AI Coach", layout="wide")
st.title("🥋 Taekwondo AI Coach")

models, preprocessing, device = load_models()

st.sidebar.header("Configuration")
target_kick = st.sidebar.selectbox("Select Drill:", TARGET_KICKS)
is_right_leg = st.sidebar.toggle("Right Leg Kick?", value=True)

if is_right_leg:
    st.sidebar.caption("Currently tracking: **Right Leg**")
else:
    st.sidebar.caption("Currently tracking: **Left Leg**")

# Show loaded model information
if target_kick in preprocessing:
    loaded_model = models[target_kick].__class__.__name__
    st.sidebar.success(f"Loaded model: {loaded_model}")

tab1, tab2 = st.tabs(["📷 Live Camera", "📁 Upload Video"])

with tab1:
    st.markdown("### Real-Time Inference")
    st.info("Step back so your full body is visible. After the countdown, execute the kick.")
    
    col1, col2 = st.columns([1, 4])
    start_cam = col1.button("🟢 Start Camera", key="start_cam")
    live_window = st.empty()

    if start_cam:
        if target_kick not in models:
            st.error(f"Model for {target_kick} is not available.")
        else:
            with st.spinner("Connecting to webcam..."):
                run_analysis(
                    video_source=0,
                    target_kick=target_kick,
                    is_right_leg=is_right_leg,
                    models=models,
                    preprocessing=preprocessing,
                    device=device,
                    display_placeholder=live_window,
                )

with tab2:
    st.markdown("### Batch / Offline Analysis")
    uploaded_video = st.file_uploader("Upload a 60FPS MP4", type=["mp4", "mov"])

    if uploaded_video is not None:
        video_window = st.empty()
        analyze_button = st.button("🚀 Analyze Uploaded Video")

        if analyze_button:
            if target_kick not in models:
                st.error(f"Model for {target_kick} is not available.")
            else:
                tfile = None
                try:
                    with st.spinner("Extracting biomechanics..."):
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(uploaded_video.read())
                        tfile.close()

                        run_analysis(
                            video_source=tfile.name,
                            target_kick=target_kick,
                            is_right_leg=is_right_leg,
                            models=models,
                            preprocessing=preprocessing,
                            device=device,
                            display_placeholder=video_window,
                        )
                finally:
                    if tfile is not None and os.path.exists(tfile.name):
                        os.unlink(tfile.name)