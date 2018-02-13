#ifndef SKEPU_CONVOLUTIONAL_H
#define SKEPU_CONVOLUTIONAL_H

#include "skepu_base.hpp"

namespace caffe
{
	class ConvolutionParameter;
}

class SkePU_Convolutional: public SkePU
{
	public:
		SkePU_Convolutional(const shared_ptr<skepu2::BackendSpec>& spec);
		//SkePU_Convolutional() = default;

		/**
		 * This functions runs the forward function of a convolutional layer
		 * Note: it takes the vector of weights rather than just one matrix of weights.
		 * returns the restructred(im2col) input data.
		 */
		void forward(DataPackage& input, DataPackage& weights, DataPackage& output, 
				const ActivationType& at, const caffe::ConvolutionParameter& conv_param,
				size_t nr_neurons, skepu2::Vector<double>& bias_weights, 
				double bias_values, bool bias = false);
		/**
		 * Calculates error according to formula:
		 * @input from the bottom layer
		 */
		void backward(DataPackage& input, DataPackage& weights, DataPackage& delta_top,
				DataPackage& delta_bottom, const ActivationType& at, 
				const caffe::ConvolutionParameter& conv_param, size_t nr_neurons);

		DataPackage _conv_in; //restructured data for forward convolution
	private:

		/**
		 * AKA im2col. Will restructred data so that the convolutional layer accesses data 
		 * consequtively. There may be duplication of data.
		 */
		void restruct_data(DataPackage& input, DataPackage& conv_in) const; 

		/**
		 * will pad the delta package so that convolution on it is possible
		 */
		void pad_delta(skepu2::Vector<size_t>& input_dims, DataPackage& delta_top, DataPackage& backprop_conv_in) const;

		/**
		 * will test copy data and measure time. impements all the backends
		 */
		void copy_image(DataPackage& conv_in, 
                DataPackage& input,
                size_t kernel_size,
                DataPackage& weights,
                size_t stride,
                size_t nr_neurons,
                size_t col_k2,
                size_t col_nrn_k2,
                size_t i);

		/*
		 * NOTE: both conv_in  and conv_out can be found in skepu_base.hpp
		 * data packaged used for data restructure one image at the time
		 * size: kernel_size ^ 2, color channels, nr_neurons/nr_kernels, batch size
		 * size is set in the forward function
		 */
		//DataPackage conv_in;

		DataPackage _conv_in_delta;

		//dimensions of one image restructure
		//skepu2::Vector<size_t> _conv_in_1_image;
};
#endif
