import sys
import os.path
import string
import parser 
import line_plotter
import bar_plotter
#import caffe_parser
import numpy as np

if len(sys.argv) < 2:

    sys.exit("Usage:" + str( sys.argv[0]) +  '[files]')

nr_files = len(sys.argv) - 1
data_files = []
path_files = None
if True:
    path_files = os.path.dirname(sys.argv[1]) #assuming all files are found in the same directory
    print 'path to save graphs to', path_files

print 'files gathered data from:',sys.argv

for i in range(nr_files):
    if '.log' in sys.argv[i+1]:
        data_files.append(parser.Parser(sys.argv[i+1]))

print 'now plotting'
#first times are caffe times
training_times = []#[66, 0, 0]
labels = ''#'caffe skepu_cpy_s memcpy_s ' #for the graphs and to differ between output in testing and training
avgs_input = []#[0.00005, 0.000040, 0.000010]
std_in =[]# [0, 0, 0]
avgs_output =[]# [0.00019, 0, 0]
std_out=[]# [0, 0, 0]
for data in data_files:
    print 'processing data for', data.name
    training_times.append(data.training_completed)
    labels += data.name + ' '

    #forward times
    #input layer
    #fts = []
    nr_layers = data.layers
    f_times = np.array(data.forward_times[0::nr_layers])
    print f_times
    print 'max ', max(f_times)
    print 'min ', min(f_times)
    #line_plotter.simple_plot(f_times, 'input layer time over iterations', 'itr', 'sec')
    avgs_input.append(np.mean(f_times))
    std_in.append(np.std(f_times))

    #output layer
    nr_layers = data.layers
    f_times = np.array(data.forward_times[7::nr_layers])
    print f_times
    print 'max ', max(f_times)
    print 'min ', min(f_times)
    #line_plotter.simple_plot(f_times, 'output layer time over iterations', 'itr', 'sec')
    avgs_output.append(np.mean(f_times))
    std_out.append(np.std(f_times))
    
    labels+='softmax '
    sm_times = np.array(data.softmax_times)
    avgs_output.append(np.mean(sm_times))
    std_out.append(np.std(sm_times))

    labels+='cross_entropy '
    ce_times = np.array(data.cross_entropy_times)
    avgs_output.append(np.mean(ce_times))
    std_out.append(np.std(sm_times))
    

bar_plotter.plot(avgs_input, 'input_layer', 'seconds', labels, std_in, path_files) 
bar_plotter.plot(avgs_output, 'output_layer', 'seconds', labels, std_out, path_files) 
bar_plotter.plot(training_times, 'training_times', 'seconds', labels, None, path_files) 


