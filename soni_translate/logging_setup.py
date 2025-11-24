from IPython import get_ipython
import logging
import sys
import warnings
import os

# Define consistent log format
LOG_FORMAT = "[%(levelname).1s] %(asctime)s >> %(message)s"
LOG_DATE_FORMAT = "%y%m%d %H:%M"


def configure_logging_libs(debug=False, name_log="sonitranslate"):
    warnings.filterwarnings(action="ignore", category=UserWarning, module="pyannote")
    modules = [
        "numba",
        "httpx",
        "markdown_it",
        "speechbrain",
        "fairseq",
        "pyannote",
        "faiss",
        "pytorch_lightning.utilities.migration.utils",
        "pytorch_lightning.utilities.migration",
        "pytorch_lightning",
        "lightning",
        "lightning.pytorch.utilities.migration.utils",
    ]
    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" if not debug else "1"

        # fix verbose pyannote audio
        def fix_verbose_pyannote(*args, what=""):
            pass

        import pyannote.audio.core.model  # noqa

        pyannote.audio.core.model.check_version = fix_verbose_pyannote
    except Exception as error:
        logging.getLogger(name_log).error(str(error))


def setup_logger(name_log, log_dir=None):
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.DEBUG)  # Always capture everything internally

    # Detect if running in a notebook
    in_notebook = True if get_ipython() is not None else False

    # Stream handler (for console)
    stream = sys.stdout if in_notebook else sys.stderr
    stream_handler = logging.StreamHandler(stream)
    stream_handler.flush = stream.flush
    stream_handler.setLevel(logging.INFO)  # Console shows INFO+ by default
    # Stream handler (for console)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    # File handler (for persistent logs) - only if log_dir provided
    file_handler = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{name_log}.log")
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture everything
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )

    # Apply handlers
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    if file_handler:
        logger.addHandler(file_handler)
    logger.propagate = False

    # configure logging libs
    configure_logging_libs(name_log=name_log)
    
    return logger


def set_logging_level(verbosity_level):
    """Change console verbosity without affecting file logging."""
    logging_level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = logging_level_mapping.get(verbosity_level, logging.INFO)

    # Update only the console handler
    for handler in logger.handlers:
        # logging.FileHandler inherits from logging.StreamHandler
        is_stream_handler = isinstance(
            handler, logging.StreamHandler
        ) and not isinstance(handler, logging.FileHandler)
        if is_stream_handler:
            handler.setLevel(level)

    logger.info(f"Console logging level set to {verbosity_level.upper()}")


def switch_log_file(new_name, log_dir=None):
    """Redirect the existing logger to a new file while keeping all handlers."""
    if log_dir:

        # Remove old file handlers
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

        # Add new file handler
        file_handler = None
        os.makedirs(log_dir, exist_ok=True)
        new_path = os.path.join(log_dir, f"{new_name}.log")
        file_handler = logging.FileHandler(new_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
    if file_handler:
        logger.addHandler(file_handler)
        logger.info(f"Switched log file to: {new_path}")
    else:
        logger.info("Switch log file skipped.")


logger = setup_logger("sonitranslate")
