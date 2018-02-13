#!/usr/bin/env python
#this scirpt will plot a single run

import sys
import os.path
import string
import parser
import line_plotter
import bar_plotter
import caffe_parser
import numpy

if len(sys.argv) < 1:
    sys.exit("How to use: \n%s [input file] [caffe directory]" %sys.argv[0])

#directory handling
data_file = sys.argv[1];
basename = os.path.basename(data_file)
out_file = basename
no_filename = False
if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
    out_file += "out.pdf"
    no_filename = True


data = parser.Parser(data_file)
#read caffe files and parse properly
if len(sys.argv) == 3:
    caffe_data_dir = sys.argv[2]
    caffe_data = caffe_parser.caffe_parser()
    caffe_data.read_caffe_logs(caffe_data_dir)
else:
    caffe_data = None
print "Now plotting..."
hidden_layers = data.layers - 2
print "layers ", data.layers
print "Hidden layers ",hidden_layers
print 'training iterations', data.training_itrs
print 'testing iterations', data.test_itrs

###BENCHMARK COMPARISON
#bar_plotter.stack_bar([data.training_completed,data.testing_completed],
#        "Training/Testing Time",
#        "Seconds",
#        "Training Testing",
#        [caffe_data.training_times[-1], caffe_data.testing_times[-1]])
###/BENCHMARK COMPARSION

#sys.exit("early exit")
if len(data.loss) == 0:
    print "No loss was outputed, exiting early."
    sys.exit(1)
#plot loss
#if(caffe_data):
#    line_plotter.loss_plot(data.loss[2:], data.display, caffe_data.loss)
#else:
#    line_plotter.loss_plot(data.loss[2:], data.display)

print "Checking if timing adds up..."
#all forward times (exluding test iterations
print 'forward times', len(data.forward_times)
#don't divide by display for times, they are printed every iteration for sake of timing
t_lim = (-(data.layers * data.test_itrs ))#/ data.display))
print 't_lim', t_lim
if(t_lim > len(data.forward_times)):
    print 'There are less times measured than there are iterations! Will skip the time plot'
else:
    train_forward_times = data.forward_times[0:t_lim]
    test_forward_times = data.forward_times[t_lim:]
    train_f = sum(data.forward_times[0:t_lim])
    test_f = sum(data.forward_times[t_lim:])
    print 'training forward time', train_f, "; times recorded", len(data.forward_times[0:t_lim])
    print 'testing forward time', test_f,"; times recorded", len(data.forward_times[t_lim:])

    tb =(sum(data.backward_times))
    print 'backward times', len(data.backward_times)
    print 'total backward time', tb

    tu =(sum(data.update_times))
    print 'update times', len(data.update_times)
    print 'total update time', tu

    training_time = train_f + tb + tu
    print 'training time summed:', training_time, ', training time recorded:', data.training_completed
    print 'testing time summed:',  test_f, ', test time recorded:', data.testing_completed
    t_t_completed = data.training_completed + data.testing_completed
    summed_tr_te = training_time+test_f
    print 'summed times vs total time:', summed_tr_te, ' - ', t_t_completed
    print (1-summed_tr_te / t_t_completed)*100, '% of time spent on calculations other than learning'


    #ts = [train_f, test_f, tb, tu, caffe_data.training_times[-1], caffe_data.testing_times[-1],
    #        caffe_data.forward_times]
    #bar_plotter.plot(ts, 'times', 'Seconds', 'training_forward testing_forward backprop update caffe_trainging
    #        caffe_testing caffe_forward caffe_backprop')


##assuming back prop and update occur at the same step in caffe
bar_plotter.plot([tb+tu, sum(caffe_data.backward_times)], "Update + Backward for SkePU and Caffe", 'Seconds',
"Skepu Caffe")

#bar_plotter.plot_histogram(data.forward_times, 'forward times', 'bins', 'amount')
#bar_plotter.plot_histogram(data.backward_times, 'backward times', 'bins', 'amount')
#bar_plotter.plot_histogram(data.update_times, 'update times', 'bins', 'amount')

#plot guess vs ans graphs
#bar_plotter.simple_plot(data.training_guesses,
#        "Training: guess vs answer",
#        "Correct",
#        ["Training"],
#        data.training_anss, True)
#
#bar_plotter.simple_plot(data.test_guesses, 
#        "Testing: guess vs answer",
#        "Correct",
#        ['Testing'],
#        data.test_anss, True)

#plot time for different functions of each layer 
bar_plotter.plot_time(data.forward_times, data.layers, data.layer_names, "Average Forward Time per Layer",
        False, caffe_data.forward_times)

#specifically for softmax, cross entropy and correctness time plotting
#bar_plotter.plot_time( [data.softmax_times, data.cross_entropy_times, data.correctness_times,
#    data.delta_times], 4,  ['softmax', 'cross entropy', 'correctness', 'delta calculation'], 'avg time for functions')

bar_plotter.plot_time(data.backward_times, data.layers, data.layer_names, "Average Backward Time per Layer", True)
bar_plotter.plot_time(data.update_times, data.layers, data.layer_names, "Average Update Time per Layer")

#learning rate plot
#line_plotter.simple_plot(data.lr, "Learning Rate", "iterations" , r"$\alpha$")

#plot total time
bar_plotter.stack_bar([data.training_completed,data.testing_completed],
        "Training/Testing Time",
        "Seconds",
        "Training Testing")
        

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
    bar_plotter.plot([r, caffe_data.result], "Accuracy", "Ratio of correct answers", "SkePU Caffe")
else:
    sys.exit("no testing answers")

tr_tm = [data.training_completed,caffe_data.training_times[-1]]
bar_plotter.plot(tr_tm,
        "Training Time",
        "Seconds",
        "SkePU Caffe")

print 'Caffe is', tr_tm[0]/tr_tm[1], 'faster than SkePU in training.'

te_tm = [data.testing_completed, caffe_data.testing_times[-1]]
bar_plotter.plot(te_tm,
        "Testing Time",
        "Seconds",
        "SkePU Caffe")

print 'Caffe is', te_tm[0]/te_tm[1], 'faster than SkePU in testing.'

tr_sc =[te_tm[0]+tr_tm[0], te_tm[1]+te_tm[1]];
bar_plotter.plot(tr_sc,
        "Total Time",
        "Seconds",
        "SkePU Caffe")

print 'Caffe is', tr_sc[0]/tr_sc[1], 'times faster than SkePU in total runtime.'
