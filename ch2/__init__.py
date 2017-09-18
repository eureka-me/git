#! env python
# -*- coding: utf-8 -*-
# Date: 2017/09/17
# Filename: __init__.py 

__author__ = 'takutohasegawa'
__date__ = "2017/09/17"

from logging import getLogger, StreamHandler, DEBUG, FileHandler

if __name__ == "__main__":
    logger = getLogger("")
    if not logger.handlers:
        fileHandler = FileHandler('sample.log')
        fileHandler.setLevel(DEBUG)
        streamHander = StreamHandler()
        streamHander.setLevel(DEBUG)
        logger.setLevel(DEBUG)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHander)

    print "Hello World";