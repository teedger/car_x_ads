"""
Logging utility for car price prediction project
Provides consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    Set up a logger with file and console handlers

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    """
    Get a logger instance

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Logger instance
    """
    if name is None:
        # Get the name of the calling module
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else __name__

    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class
    Usage: class MyClass(LoggerMixin): ...
    """

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """
    Decorator to log function execution time

    Usage:
        @log_execution_time
        def my_function():
            ...
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.info(f"Starting {func.__name__}...")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {e}")
            raise

    return wrapper


def log_dataframe_info(df, name="DataFrame", logger=None):
    """
    Log information about a pandas DataFrame

    Args:
        df: pandas DataFrame
        name: Name to identify the DataFrame
        logger: Logger instance (optional, creates one if not provided)
    """
    if logger is None:
        logger = get_logger()

    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {list(df.columns)}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"  Missing values:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            logger.info(f"    {col}: {count} ({pct:.1f}%)")
    else:
        logger.info(f"  No missing values")


def create_progress_logger(total, description="Processing", log_every=100):
    """
    Create a simple progress logger for loops

    Args:
        total: Total number of iterations
        description: Description of the operation
        log_every: Log progress every N iterations

    Returns:
        Function to call on each iteration
    """
    logger = get_logger()
    start_time = datetime.now()

    def log_progress(current):
        if current % log_every == 0 or current == total:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = current / elapsed if elapsed > 0 else 0
            pct = (current / total) * 100

            logger.info(
                f"{description}: {current}/{total} ({pct:.1f}%) - "
                f"{rate:.1f} items/sec"
            )

    return log_progress


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import time

    # Setup logger
    logger = setup_logger(
        "car_analysis",
        log_file="test.log",
        level=logging.DEBUG
    )

    logger.info("Logger initialized successfully")

    # Test log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test execution time decorator
    @log_execution_time
    def slow_function():
        time.sleep(1)
        return "Done"

    result = slow_function()

    # Test DataFrame logging
    df = pd.DataFrame({
        'a': range(100),
        'b': range(100),
        'c': [None] * 50 + list(range(50))
    })
    log_dataframe_info(df, "Test DataFrame", logger)

    # Test progress logger
    progress = create_progress_logger(1000, "Test Progress", log_every=250)
    for i in range(1, 1001):
        time.sleep(0.001)
        progress(i)

    # Test LoggerMixin
    class TestClass(LoggerMixin):
        def do_something(self):
            self.logger.info("Doing something...")

    obj = TestClass()
    obj.do_something()

    print("\nLog file created: test.log")
