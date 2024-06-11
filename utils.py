import logging
import warnings
from contextlib import contextmanager

import torch


def norm(x):
    x = x.to(torch.float32)
    x = x / 255
    # x = 2 * x - 1
    return x


def unnorm(x):
    # x = 0.5 * (x + 1)
    x = x * 255
    x = torch.floor(x)
    x = x.to(torch.uint8)
    return x


def to_numpy(x):
    return (
        x.permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )


@contextmanager
def suppress_external_logs(level=logging.CRITICAL):
    """
    A context manager to suppress logging from external modules and warnings.
    Temporarily sets the logging level of all non-root loggers to the specified level,
    and suppresses warnings.
    """
    # Backup current logging levels
    logging_levels = {}
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if logger.level != logging.NOTSET:
            logging_levels[logger_name] = logger.level
            logger.setLevel(level)

    # Suppress warnings
    warnings_filter = warnings.catch_warnings()
    warnings_filter.__enter__()
    warnings.simplefilter("ignore")

    try:
        yield
    finally:
        # Restore original logging levels
        for logger_name, level in logging_levels.items():
            logging.getLogger(logger_name).setLevel(level)

        # Restore warnings filter
        warnings_filter.__exit__(None, None, None)
