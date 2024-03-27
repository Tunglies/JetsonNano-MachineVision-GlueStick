# from src.engine import 
from src.onnx import Onnx
from src.engine import Engine
from src.logger import logging
from threading import Thread


def thrad_one(object):
    logging.info("Staring New Thread: %s", object)
    thread = Thread(target=object)
    thread.daemon = True
    thread.start()
    return object