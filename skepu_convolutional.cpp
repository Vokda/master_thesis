#include "skepu_convolutional.hpp"
#include <milli.hpp>
#include "proto/caffe.pb.h"
using namespace caffe;

/* 
 * each user function will perform a dot product between coresponding weights and image(s if color channels
 * exists)
 */
[[skepu::userfunction]]
double conv_batch(skepu2::Index1D index, 
		const double* weight, 
		const double* input, 
		const size_t image_size,
		const double B,
		const size_t nr_neurons,
		const int at,
		const size_t nr_filter
		)
{
	//which kernel is being processed.
	size_t kernel_index = index.i % nr_neurons;

	//which filter is being used
	size_t filter_index = (index.i / nr_neurons) % nr_filter;

	double result = 0;
	for(size_t i = 0; i < image_size; ++i)
	{
		result += input[i + (image_size * kernel_index)] 
			* weight[i + (filter_index * image_size)];
	}
		
	//activation
	switch(at)
	{
		case 1: //logistic
			return 1.0/ (1.0 + exp( ( -B * result ) )); 
		case 2: //relu
			return max(result, 0.0);
		case 3: //tanh
			return 2.0 / (1.0 + exp(-2.0 * result)) - 1.0;
		default: //identity
			return result;
	}
}

/*
 *  this skepu instance is for an input image that has been reordered
 *	so that each convolution's input data is consequtive.
 */
[[skepu::instance]]
static auto skepu_forward_all_conv_ordered_input = skepu2::Map<0>(conv_batch);

/**************************************************************
 * conv with bias
 * The input to this function must be preprocessed by restruct_data in order for this 
 * function to yield proper result.
 *
 * because of how the input has been reordered the @image_size is of size
 * kernel^2 * cols * nr_neurons
 * and @image_spatial_size of size
 * kernel^2 * cols
 */
[[skepu::userfunction]]
double conv_batch_bias(skepu2::Index1D index, 
		const double* weight, 
		const double* input, 
		const double* bias_weights,
		const double bias_value,
		const size_t image_size, 
		const size_t filter_size, //total filter size
		const double B,
		const size_t nr_neurons,
		const int at, 
		const size_t nr_filter
		)
{
	//which kernel is being processed.
	size_t kernel_index = index.i % nr_neurons;

	//which filter is being used
	size_t filter_index = (index.i / nr_neurons) % nr_filter;



	double result = 0;
	for(size_t i = 0; i < image_size; ++i)
	{
		result += input[i + (image_size * kernel_index)] *
			weight[i + (filter_index * image_size)];
	}
	result += bias_weights[filter_index] * bias_value;

		
	//activation
	switch(at)
	{
		case 1: //logistic
			return 1.0/ (1.0 + exp( ( -B * result ) )); 
		case 2: //relu
			return max(result, 0.0);
		case 3: //tanh
			return 2.0 / (1.0 + exp(-2.0 * result)) - 1.0;
		default: //identity function
			return result;
	}
}

/*
 *  this skepu instance is for an input image that has been reordered
 *	so that each convolution's input data is consequtive.
 */
[[skepu::instance]]
static auto skepu_forward_all_conv_ordered_input_bias = skepu2::Map<0>(conv_batch_bias);




/*
 * Will put convolved data into the output container.
 */
[[skepu::userfunction]]
double put_in_larger(skepu2::Index1D index,
		const double* input,
		const size_t image_index,
		const size_t image_size
		)
{
	//size_t filter_index = (index.i / nr_neurons) % nr_filter;

	return input[index.i + (image_index * image_size)];
}

[[skepu::instance]]
static auto put_in_output = skepu2::Map<0>(put_in_larger);







/*///////////////////forward//////////////////////
 *
 * will perform forward convolutionl
 * First the data will be reordered so that it can accessed properly.
 * After that the data will be convolved.
 *
 * The convolution is done at one image at the time to save space, as multiple 
 * elements will be duplicated.
 * 
 */
void SkePU_Convolutional::forward(DataPackage& input, DataPackage& weights, 
		DataPackage& top, const ActivationType& at, const ConvolutionParameter& conv_param,
		const size_t nr_neurons, skepu2::Vector<double>& bias_weights, double bias_value,
		bool bias)
{
	size_t kernel_size = conv_param.kernel_size(0);
	size_t stride = conv_param.stride(0); 
	size_t image_size = input.get_image_size();
	size_t image_spatial_size = input.get_image_spatial_size();
    size_t padding = conv_param.pad_size() ? conv_param.pad(0) : 0;
	/*
	 * get number of filters
	 * this is true for only the weight datapackage
	 */
	size_t nr_filters = weights.get_batch_size(); 

#ifdef DEBUG
	cout << "elements to allocate " << kernel_size*kernel_size * input.get_dimensions()[2] * nr_neurons << endl;
	cout << "kernel_size " << kernel_size << endl;
	cout << "stride " << stride << endl;
	cout << "nr neurons " << nr_neurons << endl;
    cout << "filters " << nr_filters << endl;
    cout << "output: " << top << endl;
	cout << "input: " << input << endl;
#endif


	/*
	 * set the output image to the size of the image (happens only once)
     * 
	if(conv_out.get_image_size() != conv_in.get_image_size())
	{
		skepu2::Vector<size_t> size = top.get_dimensions();
		size[3] = 1; //one image only
		conv_out.set_dimensions(size);
	}*/

#ifdef DEBUG
	assert(weights.get_image_spatial_size() == pow(kernel_size, 2));
	cout << "reordering data for conv forward" << endl;
#endif 

    /*
     * perform convolutionl on one image at the time.
     * The output of a convolution is done via iterators
     */
	for(size_t i = 0; i < input.get_batch_size(); ++i)
    {
        //size_t image_index = _conv_in.get_image_size() * i;
        if(padding > 0)
        {
        }

        //restructure data so that convolution is simply a matrix matrix multiplication (sort of)
        im2col(_conv_in, input, kernel_size, weights.get_image_spatial_size(), stride, nr_neurons, i, nullptr);	

#ifdef DEBUG
        cout << "output from restructure" << endl;
        cout << _conv_in << endl;
        press_enter_to_continue();
#endif

        //index used for iterators 
        size_t filter_index = weights.get_image_size() * i;

        //vectors 
        auto& v_output = top._data;
        auto out_start = v_output.begin() + i * top.get_image_size();
        auto out_end = out_start + top.get_image_size();
        auto& v_weights = weights._data;
        auto& v_input = _conv_in._data;

#ifdef DEBUG
        cout << "image to be convolved " << i << endl;
        cout << "weights: " << weights << endl;
        cout << "input: " << _conv_in << endl;
#endif
        if(bias)
        { 
#ifdef DEBUG
            cout << "skepu conv forward with bias" << endl;
            cout << "data used:" << endl;
            cout << "bias weights: " << bias_weights.size() << endl;
            cout << "bias value: " << bias_value << endl;
            cout << "batch size " << _conv_in.get_batch_size() << endl;
#endif

            skepu_forward_all_conv_ordered_input_bias(
                    out_start, out_end, //output
                    v_weights.begin(), //weight
                    v_input.begin(),  //input
                    bias_weights.begin(), //bias_weights
                    bias_value,
                    _conv_in.get_image_size(), //kernel^2 * cols * nr_neurons
                    //conv_in.get_image_spatial_size(), //kernel^2 * cols
                    kernel_size*kernel_size,
                    1.0f, 
                    nr_neurons,
                    at,
                    nr_filters
                    );
            //top.get_image_size());
        }
        else
        {
#ifdef DEBUG
            cout << "skepu conv forward, NO bias" << endl;
            throw runtime_error("conv no bias contains fatal bugs!");
#endif
            skepu_forward_all_conv_ordered_input(
                    out_start, out_end,
                    v_weights.begin(),
                    v_input.begin(), 
                    _conv_in.get_image_size(), 
                    //conv_in.get_image_spatial_size(),
                    //kernel_size*kernel_size,
                    1.0f, 
                    nr_neurons,
                    at,
                    //weights.get_image_size(),
                    nr_filters
                    //top.get_image_size()
                    );
        }

        //put data together
        //put_in_output(top._data, conv_out._data, i, image_size);
#ifdef DEBUG
        cout << "done for one image!" << endl;
        cout << top << endl;
        //press_enter_to_continue();
#endif
    } //end of loop for image copy and processing
#ifdef DEBUG
    cout << "done for entire batch!" << endl;
    cout << top << endl;
    //press_enter_to_continue();
#endif
} //end 

//////////////////////BACKWARD/////////////////////


/* BACKPROPAGATION
 * Backpropagate the delta to the layer below
 * This is done for the entire batch
 * @input refers to the output from previous layer (NOT the output from im2col)
 * @delta_top should have been passed through im2col before this.
 * in order to access weights rotated 180 degrees weights are accessed backwards.
*/
[[skepu::userfunction]]
double skepu_err_calc_batch(skepu2::Index1D index, 
		double bottom, 
		const double* weight, 
		const double* top_delta,
		const size_t image_size, 
		const size_t image_spatial_size,
		const size_t filter_size, //total filter size
		const int at,
		const size_t weight_size,
		const size_t weight_spatial_size,
		const size_t out_img_size,
		const size_t out_img_spatial_size)
{
	//which kernel is being processed.
	size_t output_index = index.i % out_img_spatial_size;

	//which filter is being used
	size_t bottom_filter_index = index.i / out_img_spatial_size;

	//which image is being processed
	size_t image_index = index.i / out_img_size;


	double result = 0;
	for(size_t i = 0, w = image_spatial_size; i < image_size; ++i, --w)
	{
		size_t filter_index = i / image_spatial_size;
		result += top_delta[i + (output_index * image_spatial_size) + (image_size * image_index)] *
			weight[(bottom_filter_index * weight_spatial_size) + (filter_index * weight_size) + (filter_index * weight_spatial_size) + w];
	}
		
	//derivation of activation
	switch(at)
	{
		case 1: //logistic
			return (bottom * (1.0 - bottom) * result);
		case 2: //relu
			return result <= 0 ? 0 : 1;
		case 3: //tanh
			return (1.0 - bottom * bottom) * result; 
		default: //no activation, no derivation
			return result;
	}
}

[[skepu::instance]]
static auto skepu_error_calculation = skepu2::Map<1>(skepu_err_calc_batch);



////////backward//////
void SkePU_Convolutional::backward(DataPackage& bottom, DataPackage& weights, 
		DataPackage& delta_top, DataPackage& delta_bottom, const ActivationType& at, 
		const ConvolutionParameter& conv_param, size_t nr_neurons)
{
#ifdef DEBUG
	cout << __FUNCTION__ << endl;
#endif
	size_t kernel_size = conv_param.kernel_size(0);
	size_t stride = conv_param.stride_size() > 0 ? conv_param.stride(0) : 1;
	size_t padding =  conv_param.pad_size() ? conv_param.pad(0) : 0;
	size_t nr_neurons_per_row = sqrt(nr_neurons);
#ifdef DEBUG
    cout << "bottom " << bottom << endl;
	cout << "delta top " << delta_top << endl;
    cout << "kernel size " << kernel_size << endl;
    cout << "stride " << stride << endl;
    cout << "padding " << padding << endl;
    cout << "nr_neurons_per_row " << nr_neurons_per_row << endl;
	cout << "padding data from " << delta_top._name << " to " << _padded_data._name << endl;
#endif

	////////////PAD DATA/////////////
	pad_for_output(_padded_data, 
			delta_top, 
            delta_bottom,
            weights.get_batch_size(),
            kernel_size,
            stride,
            padding);

	/* sequential padding
	for(size_t row = 0; row < d_height; ++row)
	{
		for(size_t i = 0; i < d_width; ++i)
		{
			padded_data._data[index_start + row * padded_size + i] = delta_top._data[i + row * d_width];
		}
	}
	*/

	///////////REORDER DATA////////////
    size_t bottom_nr_neurons = bottom.get_image_spatial_size();
    im2col(
            _conv_in_delta,
            _padded_data,
            kernel_size,
            weights.get_image_spatial_size(),
            stride,
            bottom_nr_neurons,
            0);

#ifdef DEBUG
	cout << "reorder data so that it can be used in a matrix matrix multiplication." << endl;
	cout << "padded data " << _padded_data << endl;
	cout << "reordered input " << _conv_in_delta << endl;
	//cout << _padded_data._name << " -> " << _conv_in_delta._name << endl;
    press_enter_to_continue();
#endif
	//now perform convolution as with forward() but with weights rotated 180d
	/*
	 * calculations needed by skepu_data_restruct() that each individual skepu functions 
	 * does not need to do.
	 */
	size_t col_k2 = _padded_data.get_dimensions()[2] * weights.get_image_spatial_size();
	size_t col_nrn_k2 = col_k2 * nr_neurons;


	///////////////ERROR CALC/////////////
	skepu_error_calculation(delta_bottom._data, 
			bottom._data,
			weights._data,
			_conv_in_delta._data,
			_conv_in_delta.get_image_size(),
			_conv_in_delta.get_image_spatial_size(), 
			kernel_size*kernel_size,
			at,
			weights.get_image_size(),
			weights.get_image_spatial_size(),
			delta_bottom.get_image_size(),
			delta_bottom.get_image_spatial_size());

#ifdef DEBUG
	cout << "performing error calculations" << endl;
	cout << "bottom: " << bottom << endl;
	cout << "weights: " << weights << endl;
	cout << "input: " << _conv_in_delta << endl;
	cout << "output: " << delta_bottom << endl;
#endif
}

//////////////////////////COPY IMAGE/////////////////////////
void SkePU_Convolutional::copy_image(DataPackage& conv_in,
        DataPackage& input,
        size_t kernel_size, 
		DataPackage& weights,
        size_t stride,
        size_t nr_neurons,
        size_t col_k2,
        size_t col_nrn_k2,
        size_t i)
{
	cout << "Convolution is done for one image at the time." << endl;
	//cout << "output for entire batch: " << top << endl;
	cout << "output (one image): " << conv_in << endl;
/*
	cout << "cpu copy." << endl;
	skepu2::BackendSpec temp_spec(skepu2::Backend::typeFromString("cpu"));
	double start = milli::GetSeconds();
	im2col(conv_in, input, kernel_size, weights.get_image_spatial_size(), stride, nr_neurons,i, temp_spec);
	double end = milli::GetSeconds();
	cout << "cpu copy time: " << end - start << " seconds." << endl;

	cout << "openmp copy." << endl;
	temp_spec = skepu2::BackendSpec(skepu2::Backend::typeFromString("openmp"));
	start = milli::GetSeconds();
	im2col(conv_in, input, kernel_size, weights.get_image_spatial_size(), stride, nr_neurons, i,temp_spec);
	end = milli::GetSeconds();
	cout << "openmp copy time: " << end - start << " seconds." << endl;

	cout << "opencl copy." << endl;
	temp_spec = skepu2::BackendSpec(skepu2::Backend::typeFromString("opencl"));
	start = milli::GetSeconds();
	im2col(conv_in, input, kernel_size, weights.get_image_spatial_size(), stride, nr_neurons, i,temp_spec);
	end = milli::GetSeconds();
	cout << "opencl copy time: " << end - start << " seconds." << endl;

	cout << "cuda copy." << endl;
	temp_spec = skepu2::BackendSpec(skepu2::Backend::typeFromString("cuda"));
	start = milli::GetSeconds();
	im2col(conv_in, input, kernel_size, weights.get_image_spatial_size(), stride, nr_neurons, i,temp_spec);
	end = milli::GetSeconds();
	cout << "cuda copy time: " << end - start << " seconds." << endl;
    */
}

/*
 *	CONSTRUCTOR
 */

SkePU_Convolutional::SkePU_Convolutional(const shared_ptr<skepu2::BackendSpec>& spec):
	SkePU(spec), 
    _conv_in_delta("convolution input delta"),
    _conv_in("conv_in")
{
	auto& be = * spec;
	skepu_error_calculation.setBackend(be);
	skepu_forward_all_conv_ordered_input.setBackend(be);
	skepu_forward_all_conv_ordered_input_bias.setBackend(be);
	//skepu_pad.setBackend(be);
}
