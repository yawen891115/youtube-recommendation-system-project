"""YouTube recommendation project; experiment tracking stays local by default."""

import os

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
