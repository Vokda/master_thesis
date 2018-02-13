#include "convolutional_layer.hpp"
#include <typeinfo>
#include <iostream>
#include "solver.hpp"
#include <math.h>
using namespace std;



ConvolutionalLayer::ConvolutionalLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				Data& data, Mapped_data& mapped_data):
	Layer(ml, spec),
	_convolution_parameter(_layer_parameter.convolution_param()),
	_weights("convolutional weights"),
	_weights_delta("weights delta"),
    _weights_sum("weights sum"),
	_skepu_f(spec)
{
	if(_convolution_parameter.kernel_size_size() > 1)
	{
		cout << "WARNING: Layer " << _layer_parameter.name() 
			<< " does not support filters of different sizes!" << endl;
	}

	//should only be one bottom
	if(get_bottoms().size() > 1)
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many bottoms!" << endl;

	//setup bottom
	_bottom = setup_bottom(mapped_data, get_bottoms().back());

	/*	calculate if the parameters are correct by
	 *	(W - F + 2P)/S + 1 = nr of neurons that fit (square result for 2d conv). 
	 *	if odd it will not work without padding.
	 *	W = one dimension of input (assuming square input image)
	 *	F = one dimension of filter size
	 *	P = padding
	 *	S = stride
	 */
	size_t input_W = (*_bottom)[0]; //bottom_ptr->get_dimensions()[0]; //width
	size_t filter_size = _convolution_parameter.kernel_size(0);
	size_t padding =  _convolution_parameter.pad_size() ? _convolution_parameter.pad(0) : 0;
	size_t stride = _convolution_parameter.stride_size() > 0 ? _convolution_parameter.stride(0) : 1;

	if(filter_size > input_W)
	{
		cout << "width " << input_W << endl;
		cout << "filter size " << filter_size << endl;
		throw runtime_error("Filter/Kernel is larger than input!");
	}
	//nr of neurons aka spatial output size (not depth). The output size will be this value^2 * depth
	double r = (input_W  - filter_size + 2 * padding)/stride + 1;
	//check if nr_neurons contains decimals by copying float value to int thereby truncating decimals
	double neurons_per_row;

	//check to make sure the kernel will fit with the input data.
	if(modf(r, &neurons_per_row) != 0 and padding == 0)
	{
		string s("Layer "+ _layer_parameter.name() +
			 " without padding this kernel size will not fit the input data!");
		//outputs 
		cout << "result (double)r: " << r << endl;
		cout << "input width " << input_W << endl;
		cout << "filter_size " << filter_size << endl;
		cout << "padding " << padding << endl; 
		cout << "stride " << stride << endl;

		throw runtime_error(s);
	}

	//check to make sure width and height of data is equal.
	if(!_bottom->is_square())
	{
		auto& dims = _bottom->get_dimensions();
		cout << "width: " << dims[0] << endl;
		cout << "height: " << dims[1] << endl;
		string s("Layer " + _layer_parameter.name() + " of type "
				+ _layer_parameter.type() + " can only supports square input currently.");
		throw runtime_error(s);
	}

	//actual number of neurons in layer
	_nr_neurons = pow(neurons_per_row, 2);


	//initialize weights

	//weight size (filter_size * filter_size * color channels * nr_filters)
	_weights.set_dimensions(
			skepu2::Vector<size_t>
			{
				filter_size, filter_size, 
				_bottom->get_dimensions()[2], 
				_convolution_parameter.num_output()
			});
#ifdef DEBUG
	cout << _weights << endl;
#endif

	_weights_delta.set_dimensions(_weights.get_dimensions());
    _weights_sum.set_dimensions(_weights.get_dimensions());

	if(!_convolution_parameter.has_weight_filler())
	{
		initialize_weights(_weights);
	}
	else
	{
		initialize_weights(_weights, _convolution_parameter.weight_filler());
	}

	//bias setup
	if(_convolution_parameter.has_bias_filler())
	{
		_bias_weights.resize(_convolution_parameter.num_output(), 0);
		double val = _convolution_parameter.bias_filler().has_value() ? 
			_convolution_parameter.bias_filler().value() : 0;
		_bias_values = val;
		initialize_weights(_bias_weights, _convolution_parameter.bias_filler());
	}


	cout << "Number of filters: " << _convolution_parameter.num_output() << endl;
	cout << "Number of neurons in layer: " << _nr_neurons << endl;


	//initialize top (nr_neurons * 
	_output_dims = skepu2::Vector<size_t> {
		static_cast<size_t>(neurons_per_row),
		static_cast<size_t>(neurons_per_row), 
		_convolution_parameter.num_output(),
		_bottom->get_batch_size()};

	vector<string>& tops = get_tops();

	if(tops.size() > 1)
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many tops!" << endl;

	_top = setup_top(data, mapped_data, tops.front(), _output_dims);

}

/* 
 * setup defnition
 */
void ConvolutionalLayer::setup(Data& data, Mapped_data& mapped_data, shared_ptr<Layer> prev)
{
	vector<string> deltas(get_deltas());

	_previous_layer = prev;

	//setup delta vectors
	const string& bottom_delta = deltas.back(); //send to bottom layer
	const string& top_delta = deltas.front(); //receive from top layer

	//there should only be one bottom so it should work
	//size_t batch_size = get_bottom_vector().back()->get_batch_size(); 

	skepu2::Vector<size_t> delta_dims = _output_dims;
	//delta_dims[3] = 1;

	//set dimensions of delta vector
	_delta_top = setup_delta_receiver(data, mapped_data, top_delta, 
			delta_dims);

	_delta_bottom = setup_delta_sender(mapped_data, bottom_delta);

#ifdef DEBUG
	cout << _weights << endl;
	cout << "bias weight ";
	cout << _bias_weights << endl;
	cout << "bias values ";
	cout << _bias_values << endl;
#endif
}

void ConvolutionalLayer::forward(bool display)
{
	_skepu_f.forward(*_bottom, _weights, *_top, 
			_activation_type, _convolution_parameter,
			_nr_neurons, _bias_weights, _bias_values, _convolution_parameter.has_bias_filler());
}

//backward

void ConvolutionalLayer::backward()
{
	//calculate errors
	_skepu_f.backward(*_bottom, _weights, *_delta_top, *_delta_bottom,
			_activation_type, _convolution_parameter, _nr_neurons);
}

void ConvolutionalLayer::update_weights(Solver& s)
{
    DataPackage* conv_in = nullptr;
    if(_bottom->get_batch_size() == 1)
        conv_in = &_skepu_f._conv_in;
	s.update_weights_conv(_weights,
            *_delta_top,
            *_bottom,
            _weights_delta,
            _weights_sum,
            _nr_neurons, 
			_convolution_parameter.stride(0), 
			_convolution_parameter.has_bias_filler(),
            _bias_values,
            _bias_weights,
            _bias_weights_delta,
            conv_in);
}

void ConvolutionalLayer::save_state()
{
	//save_weights(_weights);
}


void ConvolutionalLayer::load_state()
{
	//load_weights(_weights);
}


void ConvolutionalLayer::test_preparation()
{
	save_state();
}
