#!/usr/bin/env python
#this scirpt will plot a single run

import sys
import matplotlib.pyplot as plt
import re
import os.path
import string
import numpy as np
import caffe_parser 
import parser
import bar_plotter
import line_plotter

if len(sys.argv) < 3:
    sys.exit("How to use: \n%s [data_file] [caffe_data_dir]" %sys.argv[0])
#set to true if plots should be saved.
save = False

#read caffe files and parse properly
caffe_data_dir = sys.argv[2]
caffe_data = caffe_parser.caffe_parser()
caffe_data.read_caffe_logs(caffe_data_dir)
#caffe_data.parse_caffe_training_log(caffe_data.train_lines)

#directory
data_file = sys.argv[1];
dirname = os.path.dirname(data_file)
print dirname

#read data files
data = parser.Parser(data_file)
print "Now plotting..."
hidden_layers = data.layers - 2
print "layers ", data.layers
print "Hidden layers ",hidden_layers

#############plot training and tsting times


#plot time for different functions of each layer 
bar_plotter.plot_time(data.forward_times, data.layers, data.layer_names, "Average Forward Time per Layer", False, dirname)
bar_plotter.plot_time(data.backward_times, data.layers, data.layer_names, "Average Backward Time per Layer", True, dirname)
bar_plotter.plot_time(data.update_times, data.layers, data.layer_names, "Average Update Time per Layer", False, dirname)


#learning rate plot
line_plotter.simple_plot(data.lr, "Learning Rate", "iterations" , r"$\alpha$", dirname)
#plot loss
line_plotter.loss_plot(data.loss, data.display, caffe_data.loss, dirname)

#testing error rate
if len(data.testing_answers) > 0:
    wrongs = 0
    rights = 0
    print len(data.testing_answers)
    for x in range(len(data.testing_answers)):
        if data.testing_answers[x] == 1:
            rights += 1
        else:
            wrongs += 1
    r = float(rights) / float(len(data.testing_answers));
    bar_plotter.simple_plot(r, "Accuracy", "Correct vs Incorrect answers", "SkePU Caffe", caffe_data.result, False, dirname)
else:
    sys.exit("no testing answers")

bar_plotter.simple_plot(data.training_completed,
        "Training Time",
        "Seconds",
        "SkePU Caffe",
        caffe_data.training_times[-1],
        False,
        dirname)
