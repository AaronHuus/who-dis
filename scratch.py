import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import tensorflow as tf

from tensorflow.core.framework import graph_pb2

graph_def = graph_pb2.GraphDef()

with open('models/20180402-114759/20180402-114759.pb', "rb") as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, input_map=None, name='')

for node in graph_def.node:
    print(node.attr)
