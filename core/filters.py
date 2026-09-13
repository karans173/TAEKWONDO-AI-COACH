import numpy as np
from scipy.signal import medfilt

class OneEuroFilter():
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.t_prev = None
        self.x_prev = None

        self.dx_prev = 0.0

    def _compute_alpha(self, dt, cutoff):
        """
        smoothing factor (alpha)
        dt: Time elapsed since the last frame
        cutoff: The dynamic cutoff frequency
        """
        tau = 1.0 / (2.0 * np.pi * cutoff)
        alpha =  1.0 / (1.0 + (tau / dt))
        return alpha

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x

        dt = t - self.t_prev

        if dt <= 0.0:
            return x

        dx = (x - self.x_prev) / dt

        alpha_d = self._compute_alpha(dt, self.d_cutoff)
        smoothed_dx = (alpha_d * dx) + ((1 - alpha_d) * self.dx_prev)

        cutoff = self.min_cutoff + (self.beta * np.abs(smoothed_dx))
        alpha = self._compute_alpha(dt, cutoff)
        smoothed_x = (alpha * x) + ((1 - alpha) * self.x_prev)

        self.t_prev = t
        self.x_prev = smoothed_x
        self.dx_prev = smoothed_dx

        return smoothed_x
    
def clean_sequence(df):
    """Unexpected jumps cleaner"""
    df_clean = df.copy()
    df_clean = df_clean.astype(float)
    df_clean = df_clean.interpolate(method='linear', limit_direction='both')

    continuous_cols = [
        'base_knee_angle', 'base_pivot_angle', 
        'striking_chamber_angle', 'striking_foot_angle', 
        'spine_angle', 'striking_ankle_y', 'base_ankle_y'
    ]
    
    for col in continuous_cols:
        df_clean[col] = medfilt(df_clean[col], kernel_size=5)

    df_clean['guard_dropped'] = df_clean['guard_dropped'].rolling(
        window=5, min_periods=1, center=True
    ).median().round()
    
    return df_clean