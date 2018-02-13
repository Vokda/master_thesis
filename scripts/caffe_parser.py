import sys
import os.path
import re

class caffe_parser:
    training_answers = []
    testing_answers = []
    loss = [] 
    avg_delta = []
    softmax = []
    entropy = []
    weights = []
    lr = []
    result = 0
    avg_out = []
    training_times = []
    testing_times = []

    train_lines = [] #data read from training log
    test_lines = [] #data read from testing long
    benchmark = []

    float_pattern = re.compile('[-]?(\d\.?\d*)+(e(-)?\d+)?')
    layer_pattern = re.compile('\a+\d')
    #times
    forward_times = []
    backward_times = []
    update_times = []


#def __init__(self):
    #self.read_caffe_logs(train_log)

    def read_caffe_logs(self, raw_data):
        print 'Gathering caffe data'
        training_file = raw_data+'.train'
        testing_file = raw_data+'.test'
        benchmark_log = os.path.dirname(raw_data)+'/benchmarks/benchmark_parsed.log'

        #training file
        print 'training data'
        try:
            with open(training_file) as f:
                self.train_lines = f.read().splitlines()
            self.train_lines.pop(0)
        except:
            print 'Error: ', training_file, 'not found'
            exit(1)

        print 'test data'
        try:
            with open(testing_file) as f:
                self.test_lines = f.read().splitlines()
            self.test_lines.pop(0)
        except:
            print 'Error: ', testing_file, 'not found'
            exit(1)

        print 'benchmark data'
        try:
            with open(benchmark_log) as f:
                self.benchmark_lines = f.read().splitlines()
            self.test_lines.pop(0)
        except:    
            print 'Error: ',benchmark_log,'not found.'
            exit(1)
    
        self.parse_training_log()
        self.parse_test_log()
        self.parse_bench_log()
        print 'Done!'

    def parse_training_log(self):
        print 'parsing training log'
        for line in self.train_lines:
            sline = line.split()
            #print "iteration %i" %float(sline[0])
            #store time
            self.training_times.append(float(sline[1]))
            #store loss
            self.loss.append(float(sline[2]))
            #store learning rate
            if(len(sline) == 4):
                self.lr.append(float(sline[3]))
        print 'done!'

    def parse_test_log(self):
        print 'parsing test log'
        for line in self.test_lines:
            sline = line.split()
            #store time
            self.testing_times.append(float(sline[1]))
            #store accuracy
            self.result =float(sline[2])
        print 'done!'

    def parse_bench_log(self):
        print 'parsing benchmark log'
        for line in self.benchmark_lines:
            sline = line.split()
            if 'forward' in line:
                self.forward_times.append(float(sline[2]))
            if 'backward' in line:
                self.backward_times.append(float(sline[2]))
