import matplotlib.pyplot as plt
import numpy as np
import math
import types

#@eq_cmp shoudl the data sets be of equal length
def plot(data,  title, ylabel, bar_names, stds=None, save_dir=None):
    if not data:
        print "simple_plot for", title, ": no data given!"
        return
    if not isinstance(data, list):
        print 'it is not a list!'
        return
    
    print "Plotting bar graph for ", title
    plt.title(title)
    plt.tight_layout(w_pad=2.0)
    
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    print data
    #plt.set_default_color_cycle()
    ns = bar_names.split()
    print ns
    r = np.arange(len(data))
    col_width = 2.0
    if stds:
        print 'std ', stds

    for i in range(len(data)):
        #`print col
        if stds:
            plt.bar(i, data[i], align='center', yerr=stds[i], error_kw=dict(ecolor='red'))# label=bar_names.split()[i], color=col)
        else:
            plt.bar(i, data[i], align='center')
        #plt.xticks(i*col_width, bar_names.split()[i], rotation=30)

    plt.xticks(np.arange(len(data)), bar_names.split(), rotation=30, ha='right')



    print "Done!"

    if save_dir:
        print 'saving plot', title,'as pdf in directory', save_dir
        plt.savefig(save_dir+"/"+title+".pdf")
        plt.show()
        plt.close()
    else:
        plt.show()

def stack_bar(data, title, ylabel, bar_names, compare_data=None, save_dir=None):
    if not data:
        print "no data given!"
        return

    #if compare_data != None and eq_cmp == True:
    #    if len(data) != len(compare_data):
    #        print "data sets of different length!"
    #        print "data", data
    #        print "compare_data", compare_data
    #        return

    print "Plotting bar graph for ", title

    plt.title(title)
    bars = len(data)
    #plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    for x in range(bars):
        if x > 0:
            bottom = sum(data[0:x])
        else:
            bottom = 0
        height = data[x] 
        plt.bar(0, height, bottom=bottom, align='center', label=bar_names.split()[x], color="br"[x])
    if compare_data:
        for x in range(bars):
            if x > 0:
                bottom = sum(compare_data[0:x])
            else:
                bottom = 0
            height = compare_data[x] 
            plt.bar(1, height, bottom=bottom, align='center', label=bar_names.split()[x], color="br"[x])

    plt.legend()
    plt.tight_layout(w_pad=1.0)
    plt.xticks(np.arange(bars), bar_names.split())

    if save_dir:
        plt.savefig(save_dir+"/"+title+".pdf")
        plt.close()
    else:
        plt.show()


def plot_time(times, layers, name_of_layers, title, backprop=False, caffe_times=None, save_dir=None):
    if not times:
        print 'no data for', title
        return
    else:
        print 'Plotting ', title

    layers_timed = layers if not backprop else layers-2

    def even_data(times):
        skipped = 0
        while(len(times) % layers != 0):
            times.pop()
            skipped += 1
        if skipped > 0:
            print "number of data points skipped because it did not fit with the number of layers: ",skipped

    avg = []
    std = []
    min_vals = []
    max_vals = []
    print "times size", len(times)
    plt.title(title)
    w = 0.35

    even_data(times)

    #arrange data for each layer
    for l in range(0, layers_timed):
        layer_times = np.array(times[l::layers_timed])
        avg.append(np.mean(layer_times))
        std.append(np.std(layer_times))
        min_vals.append(min(layer_times))
        max_vals.append(max(layer_times))


    #fts = np.mean(fts, axis=1)
    r = np.arange(layers)

    #if backpropagation remove the first and last names of the layers
    layer_ticks = [] #list(name_of_layers)
    if backprop:
        layer_ticks = list(name_of_layers[2:])
        assert(len(layer_ticks) == layers-2)
        r = np.arange(layers-2)
        avg = list(reversed(avg))
        std = list(reversed(std))
    else:
        layer_ticks = name_of_layers
    #print layer_ticks

    for i in range(len(avg)):
        print 'avg ', avg[i]
        print 'min value ', min_vals[i] #if min_vals == 1 else np.amin(min_vals)
        print 'max value ', max_vals[i] #if max_vals == 1 else np.amax(max_vals)
        print '------------'
    plt.bar(r, avg, align='center', yerr=std, error_kw=dict(ecolor='red'), width=w, label='SkePU')
    if caffe_times is not None:
        plt.bar(r+w, caffe_times, align='center', width=w, color='r', label='Caffe')
        
    plt.legend()
    #plt.bar(r+0.25, avg,  yerr=[min_vals,max_vals], error_kw=dict(ecolor='green'))
    plt.xticks(r, layer_ticks)
    plt.ylabel("Seconds")
    #plt.ylim(ymax=0.1)

    if save_dir:
        plt.savefig(save_dir+"/"+title+".pdf")
        plt.close()
    else:
        plt.show()


def plot_histogram(data, title, xlabel, ylabel):
    if len(data) < 1:
        print 'no data'
        return

    print 'max value', max(data)
    print 'min value', min(data)
    print 'avg value', np.mean(data)

    #nr_bins = math.ceil(2 * pow(len(data), 1/3))
    n, bins, patches = plt.hist(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    plt.show()
