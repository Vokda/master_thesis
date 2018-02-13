#include "pooling_layer.hpp"
#include <typeinfo>
#include <iostream>
#include "solver.hpp"
#include <math.h>
using namespace std;



PoolingLayer::PoolingLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
		Data& data, Mapped_data& mapped_data):
	Layer(ml, spec),
	_pooling_parameter(_layer_parameter.pooling_param()),
	_skepu_f(spec)
{

	cout << "Kernel size: " << _pooling_parameter.kernel_size() << endl;
	cout << "Stride: " << _pooling_parameter.stride() << endl;

	//should only be one bottom
	if(get_bottoms().size() > 1)
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many bottoms!" << endl;
	bottom_ptr = setup_bottom(mapped_data, get_bottoms().back());

	//initialize top

	//check if data is square
	if(!bottom_ptr->is_square())
	{
		stringstream ss;
		ss << "Layer " << 
				_layer_parameter.name() << 
				" of type " <<
				_layer_parameter.type() << 
				" can only support square input currently. Input was: " << 
				bottom_ptr->get_dimensions()[0] <<
				" X " <<
				bottom_ptr->get_dimensions()[1];
		throw runtime_error(ss.str());
	}


	/*
	 * input = width * height
	 * output_w = (width - kernel_size)/stride + 1
	 * output_h = (height - kernel_size)/stride + 1
	 * TODO assuming square input
	 */
	//output size for one dimension of the square data
	size_t input_W = bottom_ptr->get_dimensions()[0];
	size_t filter_size = _pooling_parameter.kernel_size();
	size_t padding = _pooling_parameter.pad();  //_pooling_parameter.pad_size() ? _convolution_parameter.pad(0) : 0;
	size_t stride = _pooling_parameter.stride(); //_pooling_parameter.stride_size() > 0 ? _convolution_parameter.stride(0) : 1;

	//size_t W = (w - _pooling_parameter.kernel_size()) / _pooling_parameter.stride() + 1; original

	//make sure filter is not bigger than input
	if(filter_size > input_W)
	{
		cout << "width " << input_W << endl;
		cout << "filter size " << filter_size << endl;
        cout << *bottom_ptr << endl;
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
	if(!bottom_ptr->is_square())
	{
		cout << "width: " << (*bottom_ptr)(0) << endl;
		cout << "height: " << (*bottom_ptr)(1) << endl;
		string s("Layer " + _layer_parameter.name() + " of type "
				+ _layer_parameter.type() + " can only supports square input currently.");
		throw runtime_error(s);
	}


	_nr_neurons = pow(r, 2);


	_output_dims = skepu2::Vector<size_t>{static_cast<size_t>(r), static_cast<size_t>(r), bottom_ptr->get_dimensions()[2], bottom_ptr->get_batch_size()};

	cout << "output dimensions: " << _output_dims << endl;

	vector<string>& tops = get_tops();

	if(tops.size() > 1)
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many tops!" << endl;

	top_ptr = setup_top(data, mapped_data, tops.front(), _output_dims);

	/*
	 * sets the correct size for the number of indicies to be saved by this layer. 
	 * It will be later used in the backpropagation of this layer.
	 */
	_skepu_f.set_nr_indicies(top_ptr->get_total_size());

	if(top_ptr->_data.size() > bottom_ptr->_data.size()) //if top is larger than bottom something is terrible wrong!
	{
		cout << top_ptr << endl;
		cout << *bottom_ptr << endl;
		throw runtime_error("Top is larger than the bottom!");
	}
}


void PoolingLayer::setup(Data& data, Mapped_data& mapped_data, shared_ptr<Layer> prev)
{
	vector<string> deltas(get_deltas());
	//_next_layer = next;
	_previous_layer = prev;


	//setup delta vectors
	const string& bottom_delta = deltas.back(); //send to bottom layer
	const string& top_delta = deltas.front(); //receive from top layer
	skepu2::Vector<size_t> delta_dims = _output_dims;
	//delta_dims[3] = 1;

	setup_delta_receiver(data, mapped_data, top_delta, _output_dims);
	setup_delta_sender(mapped_data, bottom_delta);
}


void PoolingLayer::forward(bool display)
{
	DataPackage& top = *get_only_top();

	DataPackage& bottom = *get_only_bottom();

	_skepu_f.forward(bottom, top, _pooling_parameter, _nr_neurons);
#ifdef DEBUG
	bool positive = false;
	for(auto& i: top._data)
	{
		if(i > 0)
		{
			positive = true;
			break;
		}
	}
	if(!positive)
		throw runtime_error("Output was completely 0!");
#endif
}

//backward

void PoolingLayer::backward()
{
	//delta from top layer
	DataPackage& delta_top = *get_delta_vector().front();

	//delta from this layer
	DataPackage& delta_bottom = *get_delta_vector().back();

	_skepu_f.backward(delta_bottom, delta_top, _pooling_parameter);
}

void PoolingLayer::save_state()
{
	//save_weights(_weights);
}


void PoolingLayer::load_state()
{
	//load_weights(_weights);
}


void PoolingLayer::test_preparation()
{
	save_state();
}
