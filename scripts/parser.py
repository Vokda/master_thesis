import re
import sys
import os.path as os
import traceback

class Parser:

    #regex
    #can find
    #positive and negative floating point numbers (including e^x)
    #positive and negative integers
    float_pattern = re.compile('[-]?(\d\.?\d*)+(e(-)?\d+)?')

    def __init__(self, file_name):
        print 'parsing data from file', file_name, ''
        self.file_name = file_name
        self.name = os.basename(file_name)

        #learning related data
        #CORRECT vs WRONG during training/testing
        self.training_answers = []
        self.testing_answers = []

        #values regarding machine learning
        self.loss = [] 
        self.avg_delta = []
        self.softmax = []
        self.entropy = []
        self.weights = []
        #self.#avg_input = []
        self.lr = []
        self.result = []
        self.avg_out = []

        #what was guessed vs label
        self.test_guesses = []
        self.test_anss = []
        self.training_guesses =[]
        self.training_anss = []

        #times
        self.training_times = []
        self.testing_times = []
        self.training_completed = 0
        self.testing_completed = 0
        self.forward_times = []
        self.backward_times = []
        self.update_times = []
        #specific times for output layer
        self.softmax_times = []
        self.cross_entropy_times = []
        self.correctness_times = []
        self.delta_times = []

        #network data
        self.layers = 0
        self.training_itrs = 0
        self.test_itrs = 0
        self.layer_names = []
        self.display = 0



        ######actually parse
        self.parse()
	
    #pareses data
    #unlike caffe all data is in one file.
    def parse(self):
        #read file
        with  open(self.file_name) as f:
            lines = f.read().splitlines()

        print "Gathering data..."
        float_pattern = self.float_pattern
        for line in lines:
            try:
                if "loss" in line and not ":" in line: #loss
                        if self.float_pattern.search(line) is not None:
                                self.loss += [float(float_pattern.search(line).group())]
                        elif ("-nan" in line) or ("-inf" in line):
                                #self.loss += [float("-inf")]
                                raise ValueError("-inf loss was parsed from file")
                        elif ("nan" in line) or ("inf" in line):
                                #self.loss += [float("inf")]
                                raise ValueError("inf loss was parsed from file")
                elif "learning rate" in line:
                        self.lr +=[float(float_pattern.search(line).group())]
                elif "average out delta" in line:
                        self.avg_delta +=[float(float_pattern.search(line).group())]
                elif "average output" in line:
                        self.avg_out +=[float(float_pattern.search(line).group())]
                elif "result" in line:
                        if self.float_pattern.search(line) is not None:
                                self.result +=[float(float_pattern.search(line).group())]
                elif "CORRECT" in line: 
                        if "training" in line:
                                self.training_answers += [1]
                        else:
                                self.testing_answers += [1]
                elif "WRONG" in line: 
                        if "training" in line:
                            self.training_answers += [0]
                        else:
                            self.testing_answers += [0]
                elif "softmax" in line:
                        self.softmax  += [float(i) for i in float_pattern.findall(line)]
                elif "entropy" in line:
                        self.entropy += [float(i) for i in float_pattern.findall(line)]
                elif "average weight after update " in line:
                        if float_pattern.search(line) is not None:
                            self.weights += [float(float_pattern.search(line).group())]

                #times
                elif "training time" in line or "traning time" in line:
                        self.training_times += [float(float_pattern.search(line).group())]
                elif "testing time" in line:
                        self.testing_times += [float(float_pattern.search(line).group())]
                elif "forward time" in line:
                        self.forward_times += [float(float_pattern.search(line).group())]
                elif "backward time" in line:
                        self.backward_times += [float(float_pattern.search(line).group())]
                elif "update time" in line:
                        self.update_times += [float(float_pattern.search(line).group())]
                elif "Number of layers constructed" in line:
                        self.layers = int(self.float_pattern.search(line).group())
                elif "sm time" in line:
                    self.softmax_times +=[float(float_pattern.search(line).group())]
                elif "ce time" in line:
                    self.cross_entropy_times +=[float(float_pattern.search(line).group())]
                elif "cor time" in line: 
                    self.correctness_times +=[float(float_pattern.search(line).group())]
                elif "delta time" in line: 
                    self.delta_times +=[float(float_pattern.search(line).group())]

                ##guesses and answers NOT to be confused with CORRECT/WRONG answers...
                elif "test guess" in line:
                    self.test_guesses += [float(float_pattern.search(line).group())]
                elif "test correct answer" in line:
                    self.test_anss += [float(float_pattern.search(line).group())]
                elif "training guess" in line:
                    self.training_guesses += [float(float_pattern.search(line).group())]
                elif "training correct answer" in line:
                    self.training_anss += [float(float_pattern.search(line).group())]
                elif "Will train for" in line:
                    self.training_itrs = int(self.float_pattern.search(line).group())
                elif "Will test for" in line:
                    self.test_itrs = int(self.float_pattern.search(line).group())
                elif "Name of layer:" in line:
                    _, _, name = line.partition('layer: ')
                    #print line
                    #print "name ", name
                    self.layer_names.append(name)
                elif "Training time" in line:
                    self.training_completed = float(self.float_pattern.search(line).group())
                elif "Test time" in line:
                    self.testing_completed =  float(self.float_pattern.search(line).group())
                elif "Display data every" in line:
                    self.display = int(self.float_pattern.search(line).group())
            except ValueError:
                print 'Error!'
                #print self.loss
                #sys.exit(1)
            except:
                print 'line parsed:',line
                print 'Error!'
                traceback.print_exc()
                sys.exit(1)
        print "Done!"
