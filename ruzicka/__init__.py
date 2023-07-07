#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger("ruzicka")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s [%(name)s:%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
)
ch.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.propagate = False
