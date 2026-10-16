
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class RawTelemetryPacket(BaseModel):
    timestamp: float
    baro_altitude: Optional[float] = None
    geo_altitude: Optional[float] = None
    vertical_rate: Optional[float] = None
    airspeed: Optional[float] = None
    ground_speed: Optional[float] = None

class CleanTelemetryPacket(BaseModel):
    timestamp: float
    baro_altitude: float
    geo_altitude: float
    vertical_rate: float
    airspeed: float
    ground_speed: float
    imputed_fields: List[str] = Field(default_factory=list)

class KalmanStateEstimate(BaseModel):
    est_altitude: float
    est_vz: float
    est_tas: float
    est_wind: float

class SubspaceDiagnostics(BaseModel):
    t2_score: float
    t2_limit_99: float
    q_score: float
    q_limit_99: float
    is_t2_violation: bool
    is_q_violation: bool

class SensorContribution(BaseModel):
    sensor: str
    contribution_pct: float
    raw_residual: float

class AlertNotification(BaseModel):
    timestamp: float
    severity: str  # 'HEALTHY', 'WARNING', 'CRITICAL'
    title: str
    faulty_sensor: Optional[str] = None
    description: str
    contributions: List[SensorContribution] = Field(default_factory=list)

class DiagnosticsResult(BaseModel):
    step_id: int
    timestamp: float
    raw_telemetry: Dict[str, Optional[float]]
    clean_telemetry: Dict[str, float]
    imputed_fields: List[str]
    kalman_states: KalmanStateEstimate
    tracking_residuals: Dict[str, float]
    subspace: SubspaceDiagnostics
    alert: AlertNotification

class FaultInjectionRequest(BaseModel):
    target_sensor: str = Field(default="airspeed", description="Target sensor channel")
    fault_type: str = Field(default="drift", description="'drift', 'bias', 'scale', 'noise'")
    magnitude: float = Field(default=0.04, description="Fault magnitude or drift rate per second")
    enabled: bool = Field(default=True, description="Enable or disable fault injection")

class PipelineStatus(BaseModel):
    is_running: bool
    ingestion_source: str
    total_samples_processed: int
    baseline_calibrated: bool
    active_fault: Optional[Dict] = None
    last_alert: Optional[AlertNotification] = None
