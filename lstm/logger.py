#
# Copyright (C) 2025 Austin Raney, Lynker
#
# Author: Austin Raney <araney@lynker.com>
#
from __future__ import annotations

import logging
import logging.config

logger = logging.getLogger("bmi.lstm")


def configure_logging(level: logging._Level = logging.INFO):
    logging.config.dictConfig(logging_config)
    logger.setLevel(level)


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "incremental": False,
    "formatters": {
        "short": {
            "format": "[%(levelname)s|%(name)s|%(asctime)s]: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
        "verbose": {
            "format": "[%(levelname)s|%(name)s|%(asctime)s|%(module)s.%(funcName)s:%(lineno)d]: %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "short",
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "verbose",
            "level": "WARNING",
        },
    },
    "loggers": {
        "root": {
            "handlers": ["stdout", "stderr"],
            "level": "INFO",
            "propagate": False,
        }
    },
}
