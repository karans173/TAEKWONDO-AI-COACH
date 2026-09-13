import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import urllib.request
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.biomechanics import BiomechanicsEngine
from core.state_machine import StateMachine
from core.filters import OneEuroFilter

def extract_kick_to_csv(video_path, output_csv_path, is_right_leg=True):
    model_path = 'pose_landmarker_heavy.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Heavy Model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        urllib.request.urlretrieve(url, model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    bio_engine = BiomechanicsEngine()
    state_machine = StateMachine()
    skeleton_filter = OneEuroFilter(min_cutoff=0.8, beta=0.08, d_cutoff=1.0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 60.0 
        
    frame_count = 0
    print(f"Processing {os.path.basename(video_path)}...")
    
    last_valid_coords = np.zeros((33, 4))
    foot_indices = {27, 28, 29, 30, 31, 32}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        current_time = frame_count / fps 
        timestamp_ms = int(current_time * 1000)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks:
            raw_coords = np.zeros((33, 4))
            pose = results.pose_landmarks[0] 
            
            for i, lm in enumerate(pose):
                current_vis = getattr(lm, 'visibility', 1.0)
                
                current_val = [lm.x * w, lm.y * h, lm.z * w, current_vis]
                
                if i in foot_indices:
                    is_out_of_bounds = lm.x < 0.0 or lm.x > 1.0 or lm.y < 0.0 or lm.y > 1.0
                    is_hidden = current_vis < 0.3
                    
                    if (is_out_of_bounds or is_hidden) and frame_count > 1:
                        raw_coords[i] = last_valid_coords[i]
                    else:
                        raw_coords[i] = current_val
                        last_valid_coords[i] = current_val
                else:
                    raw_coords[i] = current_val
                    last_valid_coords[i] = current_val

            filtered_xyz = skeleton_filter(
                raw_coords[:, :3].ravel(),
                current_time
            ).reshape(-1, 3)

            filtered_landmarks = np.column_stack((
                filtered_xyz,
                raw_coords[:, 3]
            )) 
            
            features_dict = bio_engine.extract_kinematic_chains(filtered_landmarks, current_time, is_right_leg=is_right_leg)
            
            ankle_idx = 28 if is_right_leg else 27
            current_ankle = filtered_landmarks[ankle_idx][:3] 
            
            current_state, velocity, captured_seq = state_machine.update(
                features_dict, current_ankle, current_time, h
            )
            
            if captured_seq is not None:
                df = pd.DataFrame(captured_seq)
                df.to_csv(output_csv_path, index=False)
                print(f"-> Successfully Captured {len(captured_seq)} frames.")
                break

            if state_machine.current_state.name == "KICKING":
                captured_seq = list(state_machine.kick_buffer)
                final_peak = state_machine.peak_elevation
                kick_threshold = state_machine.KICK_HEIGHT_THRESHOLD * h
                
                if final_peak >= kick_threshold and len(captured_seq) >= 15:
                    df = pd.DataFrame(captured_seq)
                    df.to_csv(output_csv_path, index=False)
                    print(f"-> Successfully Captured {len(captured_seq)} frames (Forced flush at EOF).")
                else:
                    print("-> Skipped at EOF: Incomplete kick or shuffle.")

    cap.release()

if __name__ == "__main__":
    # Test block
    extract_kick_to_csv("data/dataset/frontkick_errorNoRetraction_02_cam45.mp4", 
                        "data/frontkick_errorNoRetraction_02_cam45.csv",
                        is_right_leg=True)