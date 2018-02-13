import numpy as np
import matplotlib.pyplot as plt

def simple_plot(data, title, xlabel, ylabel, save_dir=None):
    if not data.any():
        print "simple plot: no data"
        return

    print "Plotting ",title
    print "data size", len(data)
    plt.title(title)
    r = np.arange(0, len(data))
    plt.plot(r, data, color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_dir:
        plt.savefig(save_dir+"/"+title+".pdf")
        plt.close()
    else:
        plt.show()

def loss_plot(loss, display, caffe_loss=None, save_dir=None):
    if not loss:
        print "no loss data"
        return
    if not caffe_loss:
        print "no caffe data"

    print "Plotting Loss"
    loss_length = len(loss)*display
    if caffe_loss:
        caffe_length = len(caffe_loss)*display
        if caffe_loss != loss_length:
            print "different amount of data for losses: ", loss_length, caffe_length
            #loss.pop(0)
            #loss.pop(0)
            loss_length = len(loss)*display

        r = np.arange(caffe_length, step=display)
        plt.plot(r, caffe_loss, 'r', label='Caffe Loss')

    r = np.arange(loss_length, step=display)
    plt.plot(r, loss, 'b', label='Loss')
    plt.xlabel("iterations")

    if not caffe_loss:
        plt.title("Loss")
    else:
        plt.title("Loss vs Caffe")

    plt.ylabel("loss")
    plt.legend()

    if save_dir:
        plt.savefig(save_dir+"/loss.pdf")
        plt.close()
    else:
        plt.show()
