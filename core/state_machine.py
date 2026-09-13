# core/state_machine.py
import numpy as np
from enum import Enum
from collections import deque

class KickState(Enum):
    STANCE = 1
    KICKING = 2

class StateMachine():
    def __init__(self):
        self.current_state = KickState.STANCE
        self.kick_buffer = []
        
        self.pre_kick_history = deque(maxlen=10)

        self.prev_time = None
        self.prev_ankle_pos = None
        self.resting_y = None
        
        self.peak_elevation = 0.0

        self.LIFT_OFF_THRESHOLD = 0.03       # Lift 3% to start recording
        self.KICK_HEIGHT_THRESHOLD = 0.15    # Foot must lift at least 15% of screen height

    def update(self, features_dict, current_ankle_pos, current_time, frame_height=1.0):
        
        lift_threshold = self.LIFT_OFF_THRESHOLD * frame_height
        kick_threshold = self.KICK_HEIGHT_THRESHOLD * frame_height
        
        velocity = 0.0
        if self.prev_ankle_pos is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                dist = np.linalg.norm(current_ankle_pos - self.prev_ankle_pos)
                velocity = dist / dt
                
        self.prev_ankle_pos = current_ankle_pos
        self.prev_time = current_time

        striking_y = features_dict["striking_ankle_y"]


        if self.current_state == KickState.STANCE:
            
            self.pre_kick_history.append(features_dict)
            
            if self.resting_y is None:
                self.resting_y = striking_y
            else:
                self.resting_y = max(self.resting_y, striking_y)

            foot_lifted = striking_y < (self.resting_y - lift_threshold)

            if foot_lifted:
                self.current_state = KickState.KICKING
                
                self.kick_buffer = list(self.pre_kick_history) 
                
                self.peak_elevation = 0.0 
                print(f"STANCE -> KICKING (Action Started - Included {len(self.kick_buffer)} pre-frames)")

        elif self.current_state == KickState.KICKING:
            self.kick_buffer.append(features_dict)
            
            current_elevation = self.resting_y - striking_y
            self.peak_elevation = max(self.peak_elevation, current_elevation)
            

            if self.peak_elevation >= kick_threshold:
                active_landing_threshold = 0.08 * frame_height
            else:
                active_landing_threshold = lift_threshold
                
            foot_on_ground = striking_y >= (self.resting_y - active_landing_threshold)
            
            if foot_on_ground:
                self.current_state = KickState.STANCE
                captured_seq = list(self.kick_buffer)
                
                final_peak = self.peak_elevation
                
                # Reset for the next movement
                self.kick_buffer = []
                self.pre_kick_history.clear() 
                self.resting_y = None 
                self.peak_elevation = 0.0
                
                if final_peak >= kick_threshold:
                    print(f"KICKING -> STANCE (Success! Height: {final_peak:.1f}, Frames: {len(captured_seq)})")
                    return self.current_state, velocity, captured_seq
                else:
                    print(f"Ignored: Walking/Shuffling (Height only {final_peak:.1f})")
                    return self.current_state, velocity, None

        return self.current_state, velocity, None