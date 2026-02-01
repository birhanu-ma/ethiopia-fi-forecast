import logging
import os

# Ensure a logs folder exists
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "eda_enrichment.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",                  # Append mode
    level=logging.INFO,            # Capture INFO, WARNING, ERROR
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Optional: also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Logging Initialized")
