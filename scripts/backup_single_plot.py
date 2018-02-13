#!/usr/bin/env python
#this scirpt will plot a single run

import sys
import matplotlib.pyplot as plt
import os.path
import string

if len(sys.argv) < 2:
    sys.exit("How to use: \n%s network_directory/" %sys.argv[0])

#directory
network_directory = sys.argv[1];

#file names
train_data ="train_delta"
test_data = "test_delta"
time_data = "construction_time"

#the plot object
plot_obj

def plot_delta(in_data, plt):
    in_file = network_directory + in_data + ".dat"
    out_file = network_directory + in_data + ".pdf"

    print "Input file: %s" %in_file
    print "Output file: %s" %out_file

    y_label = "Average Delta for output layer"

    title = "Average error"
    plot_label = ""
    if "train" in in_data:
        title += " for training set"
        plot_label = "Average training error"
    else:
        title += " for test set"
        plot_label = "Average testing error"

    data_file = open(in_file)
    line = data_file.readline()
    data_vector = [float(x) for x in line.split()]

    #fig1 = plt.figure()

    # Plotting display settings
    plt.title(title)
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel("Iteration", fontsize=18)

    plt.plot(data_vector[:], label=plot_label)

    plt.legend(loc='upper left')
    # Show plot
    plt.show()
    # Save plot
    #plt.savefig(out_file)

plot_delta(train_data, plot_obj)
plot_delta(test_data, plot_obj)
plot_obj.show()
