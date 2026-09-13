# core/biomechanics.py
import numpy as np

class BiomechanicsEngine:
    def __init__(self):
        self.baseline_foot_vector_xz = None

    def reset_baseline(self):
        """
        Resets baseline foot vector for a new kick.
        """
        self.baseline_foot_vector_xz = None

    def calculate_angle_3d(self, a, b, c):
        """
        Calculates the 3D angle between three points where 'b' is the vertex.
        """
        BA = a - b
        BC = c - b
        
        magnitude_ba = np.linalg.norm(BA)
        magnitude_bc = np.linalg.norm(BC)
        
        if magnitude_ba == 0.0 or magnitude_bc == 0.0:
            return 0.0
            
        cosine_angle = np.dot(BA, BC) / (magnitude_ba * magnitude_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        return float(np.degrees(np.arccos(cosine_angle)))
    
    def extract_kinematic_chains(self, landmarks, current_time=None, is_right_leg=True):
        if is_right_leg:
            s_hip, s_knee, s_ankle, s_toe = 24, 26, 28, 32 # Striking right leg
            b_hip, b_knee, b_ankle, b_heel, b_toe = 23, 25, 27, 29, 31 # Base left leg
            guard_idx, hip_idx = 15, 23 # Left hand guards right leg kick
        else:
            s_hip, s_knee, s_ankle, s_toe = 23, 25, 27, 31 # Striking left leg
            b_hip, b_knee, b_ankle, b_heel, b_toe = 24, 26, 28, 30, 32 # Base right leg
            guard_idx, hip_idx = 16, 24 # Right hand guards left leg side kick
            
        def get_pt(idx):
            return landmarks[idx][:3]

        # 1. Base / Pivot Angle
        base_knee_angle = self.calculate_angle_3d(get_pt(b_hip), get_pt(b_knee), get_pt(b_ankle))
        
        heel_xz = np.array([landmarks[b_heel][0], 0.0, landmarks[b_heel][2]])
        toe_xz = np.array([landmarks[b_toe][0], 0.0, landmarks[b_toe][2]])
        current_foot_vector = toe_xz - heel_xz
        mag_current = np.linalg.norm(current_foot_vector)
        
        if self.baseline_foot_vector_xz is None:
            if mag_current > 0.0:
                self.baseline_foot_vector_xz = current_foot_vector
            base_pivot_angle = 0.0
        else:
            mag_baseline = np.linalg.norm(self.baseline_foot_vector_xz)
            if mag_baseline == 0.0 or mag_current == 0.0:
                base_pivot_angle = 0.0
            else:
                cosine_angle = np.dot(current_foot_vector, self.baseline_foot_vector_xz) / (mag_baseline * mag_current)
                base_pivot_angle = float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

        # 2. Striking Limb Angles
        striking_chamber_angle = self.calculate_angle_3d(get_pt(s_hip), get_pt(s_knee), get_pt(s_ankle))
        striking_foot_angle = self.calculate_angle_3d(get_pt(s_knee), get_pt(s_ankle), get_pt(s_toe))

        # 3. Spine / Posture
        shoulder_mid = (get_pt(11) + get_pt(12)) / 2.0
        hip_mid = (get_pt(23) + get_pt(24)) / 2.0
        spine_vector = shoulder_mid - hip_mid
        
        y_axis = np.array([0.0, 1.0, 0.0])
        spine_mag = np.linalg.norm(spine_vector)
        if spine_mag == 0.0:
            spine_angle = 0.0
        else:
            spine_cos = np.dot(spine_vector, y_axis) / spine_mag
            spine_angle = float(np.degrees(np.arccos(np.clip(spine_cos, -1.0, 1.0))))

        # 4. Guard Check 
        guard_lm = landmarks[guard_idx]
        is_visible = guard_lm[3] > 0.4
        
        # left striking leg -> right hand on guard while other can be down, vica-versa
        if not is_visible:
            guard_dropped = False
        else:
            guard_dropped = landmarks[guard_idx][1] > landmarks[hip_idx][1]

        return {
            "base_knee_angle": round(base_knee_angle, 3),
            "base_pivot_angle": round(base_pivot_angle, 3),
            "striking_chamber_angle": round(striking_chamber_angle, 3),
            "striking_foot_angle": round(striking_foot_angle, 3),
            "spine_angle": round(spine_angle, 3),
            "striking_ankle_y": round(float(landmarks[s_ankle][1]), 4),
            "base_ankle_y": round(float(landmarks[b_ankle][1]), 4),
            "guard_dropped": bool(guard_dropped)
        }