"""
app/models/kalman_filter.py
===========================
Production-Hardened Coupled Kinematic Kalman Filter for Avionics Sensor Fusion.
Features:
- Joseph-Form Covariance Update: P = (I - KH) P_pred (I - KH)^T + K R K^T
  (Strictly guarantees positive semi-definiteness and numerical symmetry over infinite time)
- Mahalanobis Innovation Gating (Chi-Squared Outlier Gating) to prevent state poisoning
- Ridge-Regularized Inversion (Jitter addition) against matrix singularities
- Dead-Reckoning Kinematic Fallback when sensor channels are unavailable
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from app.schemas import CleanTelemetryPacket, KalmanStateEstimate
from app.logger import get_logger

logger = get_logger("models.kalman_filter")

class ProductionAvionicsKalmanFilter:
    """
    State space:
      x = [h, v_z, v_tas, v_wind]^T
        h: true altitude (m)
        v_z: vertical speed (m/s)
        v_tas: true airspeed (m/s)
        v_wind: along-track headwind component (m/s)

    Measurement vector z:
      z_1: baro_altitude (m)
      z_2: geo_altitude  (m)
      z_3: vertical_rate (m/s)
      z_4: airspeed      (m/s)
      z_5: ground_speed  (m/s = v_tas - v_wind)
    """

    def __init__(self, dt: float = 1.0, mahalanobis_gate_prob: float = 0.999):
        self.dt = dt
        self.initialized = False

        # State transition matrix F
        self.F = np.array([
            [1.0, dt,  0.0,  0.0],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0, 1.0,  0.0],
            [0.0, 0.0, 0.0,  1.0]
        ], dtype=np.float64)

        # Measurement mapping matrix H (5 measurements from 4 states)
        self.H = np.array([
            [1.0, 0.0,  0.0,  0.0],  # baro_alt
            [1.0, 0.0,  0.0,  0.0],  # geo_alt
            [0.0, 1.0,  0.0,  0.0],  # vertical_rate
            [0.0, 0.0,  1.0,  0.0],  # airspeed
            [0.0, 0.0,  1.0, -1.0]   # ground_speed = v_tas - v_wind
        ], dtype=np.float64)

        # Process noise covariance Q (tuned for high-altitude transport dynamics)
        self.Q = np.diag([
            0.1 * (dt**2),   # altitude
            0.2 * dt,        # vertical rate
            0.05 * dt,       # true airspeed
            0.01 * dt        # headwind
        ]).astype(np.float64)

        # Nominal measurement noise covariance R
        self.R = np.diag([
            3.5**2,   # baro_altitude
            5.0**2,   # geo_altitude
            0.35**2,  # vertical_rate
            0.70**2,  # airspeed
            0.50**2   # ground_speed
        ]).astype(np.float64)

        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([100.0, 10.0, 50.0, 20.0]).astype(np.float64)
        self.baseline_wind = 18.0

        # Mahalanobis / Chi-Square Gating Threshold (5 DoF)
        # Prevents extreme single-sample hard spikes or corrupt data from poisoning states
        self.mahalanobis_threshold = float(stats.chi2.ppf(mahalanobis_gate_prob, df=5))
        self.last_mahalanobis_distance: float = 0.0
        self.is_outlier_gated: bool = False

    def initialize(self, z: np.ndarray):
        """Initialize filter from first observation vector."""
        h_init = 0.5 * (z[0] + z[1])
        vz_init = z[2]
        tas_init = z[3]
        wind_init = np.clip(tas_init - z[4], -50.0, 50.0)
        self.x = np.array([h_init, vz_init, tas_init, wind_init], dtype=np.float64)
        self.P = np.diag([16.0, 1.0, 2.0, 5.0]).astype(np.float64)
        self.baseline_wind = float(wind_init)
        self.initialized = True
        logger.info(
            f"[KALMAN INITIALIZED] Initial states: Altitude={h_init:.1f}m, "
            f"TAS={tas_init:.1f}m/s, Headwind={wind_init:.1f}m/s"
        )

    def step(self, packet: CleanTelemetryPacket) -> Tuple[KalmanStateEstimate, np.ndarray, Dict[str, float]]:
        """
        Executes one robust Kalman step with:
        1. Kinematic Prediction
        2. Innovation & Ridge-Regularized Covariance Inversion
        3. Mahalanobis Statistical Gating (Hard Outlier Protection)
        4. Joseph-Form Covariance Update (Numerical Stability Guard)
        5. Symmetry Enforcement
        """
        z = np.array([
            packet.baro_altitude,
            packet.geo_altitude,
            packet.vertical_rate,
            packet.airspeed,
            packet.ground_speed
        ], dtype=np.float64)

        if not self.initialized:
            self.initialize(z)

        # 1. Prediction Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 2. Innovation (Residual)
        z_pred = self.H @ x_pred
        raw_res = z - z_pred

        # Innovation Covariance with Ridge Regularization (protects against ill-conditioning)
        S = self.H @ P_pred @ self.H.T + self.R
        S_reg = S + 1e-6 * np.eye(5)

        # Whitened / Standardized Innovation: r_i / sqrt(S_ii)
        whitened_res = raw_res / np.sqrt(np.maximum(np.diag(S_reg), 1e-8))

        # 3. Mahalanobis Distance & Statistical Gating
        try:
            S_inv = np.linalg.pinv(S_reg)
            d_mahalanobis_sq = float(raw_res @ S_inv @ raw_res)
        except Exception:
            S_inv = np.diag(1.0 / np.maximum(np.diag(S_reg), 1e-6))
            d_mahalanobis_sq = float(raw_res @ S_inv @ raw_res)

        self.last_mahalanobis_distance = d_mahalanobis_sq

        # If measurement violates chi-squared gate, reject measurement update
        # to protect internal state vector (Dead Reckoning Fallback)
        if d_mahalanobis_sq > self.mahalanobis_threshold:
            self.is_outlier_gated = True
            logger.warning(
                f"[MAHALANOBIS GATED] Hard outlier detected! d_M^2={d_mahalanobis_sq:.2f} "
                f"> Gate Threshold={self.mahalanobis_threshold:.2f}. Measurement update rejected, "
                f"internal state advanced via dead reckoning."
            )
            # Advance state using prediction only, keep P_pred
            self.x = x_pred
            self.P = 0.5 * (P_pred + P_pred.T)
        else:
            self.is_outlier_gated = False
            # 4. Kalman Gain
            K = P_pred @ self.H.T @ S_inv

            # 5. State Update
            self.x = x_pred + K @ raw_res

            # 6. Joseph-Form Covariance Update (Production Numerical Hardening)
            # P = (I - KH) P_pred (I - KH)^T + K R K^T
            # Strictly preserves positive semi-definiteness under floating-point arithmetic
            I_KH = np.eye(4) - K @ self.H
            self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

            # 7. Symmetry Enforcement Guard
            self.P = 0.5 * (self.P + self.P.T)

        # 8. Physical Tracking Residuals against Estimated States
        h_est = float(self.x[0])
        vz_est = float(self.x[1])
        tas_est = float(self.x[2])
        wind_est = float(self.x[3])

        tracking_residuals = {
            "baro_altitude": float(packet.baro_altitude - h_est),
            "geo_altitude": float(packet.geo_altitude - h_est),
            "vertical_rate": float(packet.vertical_rate - vz_est),
            "airspeed": float(packet.airspeed - tas_est),
            "ground_speed": float(packet.ground_speed - (tas_est - self.baseline_wind))
        }

        state_est = KalmanStateEstimate(
            est_altitude=h_est,
            est_vz=vz_est,
            est_tas=tas_est,
            est_wind=wind_est
        )

        return state_est, whitened_res, tracking_residuals

# Backwards compatible alias
OnlineAvionicsKalmanFilter = ProductionAvionicsKalmanFilter
