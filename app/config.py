import os
from typing import Optional,List
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
# Automatic zero-dependency .env loader
def _load_env_file(dotenv_path: str = ".env"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val

_load_env_file()

class PipelineConfig(BaseModel):
    
    INGESTION_SOURCE: str = os.getenv("INGESTION_SOURCE", "simulation")  
    SIMULATION_DT: float = 1.0 
    POLL_INTERVAL_SECONDS: float = float(os.getenv("POLL_INTERVAL_SECONDS", "1.0"))
    MAX_BUFFER_SIZE: int = int(os.getenv("MAX_BUFFER_SIZE", "1000"))

    # OpenSky Network API Credentials & Filters (https://opensky-network.org)
    
    OPENSKY_USERNAME: Optional[str] = os.getenv("OPENSKY_USERNAME", None)
    OPENSKY_PASSWORD: Optional[str] = os.getenv("OPENSKY_PASSWORD", None)
    OPENSKY_TARGET_ICAO24: Optional[str] = os.getenv("OPENSKY_TARGET_ICAO24", None)  

    # Missing Data Imputation Window
    MAX_FFILL_STALE_STEPS: int = 10 

    # Kalman Filter Parameters
    KF_STATE_DIM: int = 4
    KF_MEAS_DIM: int = 5
    KF_EWMA_LAMBDA: float = 0.15

    # PCA Subspace Residual Monitor
    PCA_N_COMPONENTS: int = 2
    PCA_CONFIDENCE_LEVEL: float = 0.99
    RESIDUAL_WINDOW_SIZE: int = 20  
    BASELINE_CALIBRATION_STEPS: int = 300  

    # Server & Networking
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Active Sensor Channels
    SENSOR_CHANNELS: List[str] = [
        "baro_altitude",
        "geo_altitude",
        "vertical_rate",
        "airspeed",
        "ground_speed"
    ]

settings = PipelineConfig()
