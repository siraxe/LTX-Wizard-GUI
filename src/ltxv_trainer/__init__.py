import logging
import sys

try:
    from accelerate.utils import is_deepspeed_zero3_enabled, is_main_process, get_process_index
    IS_MULTI_GPU = is_deepspeed_zero3_enabled() or not is_main_process()
    RANK: int = get_process_index()
except ImportError:
    # If accelerate utils are not fully available, assume not a multi-GPU environment
    # and set rank to 0 (main process)
    is_main_process = lambda: True # Define a dummy function for is_main_process
    IS_MULTI_GPU = False
    RANK: int = 0 # Default rank to 0 for single process

from loguru import logger
from rich.logging import RichHandler

# Get the rank of the current process (0 for main process)
# RANK: int = logger._core.handlers[0]._sink.level  # This might need adjustment based on loguru internals or use Accelerate\'s get_process_index() if available

if not is_main_process():
    # Remove default loguru handlers for non-main processes to avoid duplicate logs
    logger.remove()


# Configure with Rich
logging.basicConfig(
    level="INFO",
    format=f"[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

# Get the logger and configure it
logger = logging.getLogger("ltxv_trainer")
logger.setLevel(logging.DEBUG) # Keep logger level at DEBUG to capture all messages initially
logger.propagate = True # Allow logs to propagate to the root handler

# Set level based on process (only main process logs INFO/DEBUG)
if not is_main_process():
     logger.setLevel(logging.WARNING) # Non-main processes only log WARNING and above


# Expose common logging functions directly
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

# The logger instance itself is also available as 'logger'
