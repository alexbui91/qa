from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.tools import freeze_graph

from embedding_compression import Compression

import argparse
import os

import utils

class FreezeGraphTest():

    def _testFreezeGraph(self):

        checkpoint_path = "weights/compression.weights"
        input_graph_name = "input_graph.pb"
        output_graph_name = "output_graph.pb"
        const_prefix = "const"
        model = Compression()
        model.init_opts()
        tconfig = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tconfig) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            graph_io.write_graph(sess.graph, const_prefix, input_graph_name)

        # We save out the graph to disk, and then call the const conversion
        # routine.
        input_graph_path = os.path.join(const_prefix, input_graph_name)
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "Sum"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_graph_path = os.path.join(const_prefix, output_graph_name)
        clear_devices = False

        freeze_graph.freeze_graph(
            input_graph_path, input_saver_def_path, input_binary, checkpoint_path,
            output_node_names, restore_op_name, filename_tensor_name,
            output_graph_path, clear_devices, "", "", checkpoint_path)

        # Now we make sure the variable is now a constant, and that the graph still
        # produces the expected result.
        with ops.Graph().as_default():
            output_graph_def = graph_pb2.GraphDef()
            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = importer.import_graph_def(output_graph_def, name="")

            for node in output_graph_def.node:
                if node.name == "one_hot":
                    print(node.attr['value'].tensor)

            # with tf.Session(config=tconfig) as sess:
            #     output_node = sess.graph.get_tensor_by_name("one_hot")
            #     print

    def get_trained(self, layer="", url="", prefix=""):
        checkpoint_meta = "%s/%s_compression.weights.meta" % (url, prefix)
        checkpoint = "%s/%s_compression.weights" % (url, prefix)
        model = Compression()
        model.init_opts()
        tconfig = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tconfig) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.import_meta_graph(checkpoint_meta)
            # saver.restore(sess, tf.train.latest_checkpoint(url))
            saver.restore(sess, checkpoint)
            val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer)
            # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            if val:
                val = val[0]
                return val.eval()

if __name__ == "__main__":
    # test.main()
    parse = argparse.ArgumentParser()
    parse.add_argument("-u", '--url')
    parse.add_argument("-u2", '--url2')
    parse.add_argument("-p", '--prefix')
    parse.add_argument("-f", '--folder', type=int, default=0)


    args = parse.parse_args()

    fc = FreezeGraphTest()
    # fc._testFreezeGraph()
    if args.folder:
        files = os.listdir(args.url)
        name = []
        for f in files:
            if 'compression.weights' in f:
                a = f.split('.')[0]
                a = a.split('_')[0]
                if a not in name:
                    name.append(a)
        path = args.url
        if args.url2:
            path = args.url2
        for n in name:
            code_book = fc.get_trained("code_book", args.url, n)
            utils.save_file("%s/%s_code_book.pkl" % (path, n), code_book)
            tf.reset_default_graph()
    else:
        code_book = fc.get_trained("code_book", args.url, args.prefix)
        utils.save_file("%s/%s_code_book.pkl" % (args.url, args.prefix), code_book)
