#include "fully_connected_layer.hpp"
#include <typeinfo>
#include <iostream>
#include "solver.hpp"
#include "parser.hpp"
using namespace std;



FullyConnectedLayer::FullyConnectedLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				Data& data, Mapped_data& mapped_data):
	Layer(ml, spec),
	_inner_product_parameter(_layer_parameter.inner_product_param()),
	_weights(_inner_product_parameter.num_output(),1), _weights_delta(_weights),
	_skepu_f(spec)// ,
    //_bias_weights(1,-1), _bias_weights_delta(1,-1)
{
	//initialize bottom

	//should only be one bottom 
	if(get_bottoms().size() > 1) 
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many bottoms!" << endl; 
	const string& bottom_name = get_bottoms().back();
	_bottom = setup_bottom(mapped_data, bottom_name);

	//initialize weights 
	size_t image_size = _bottom->get_image_size();
	cout << "Number of weights per neuron: " << image_size << endl; 
	_weights.resize(_inner_product_parameter.num_output(), image_size, 0);
	_weights_delta.resize(_inner_product_parameter.num_output(), image_size, 0); //should be zero since there has been no change since "previous" iteration.
	
	
	if(!_inner_product_parameter.has_weight_filler())
	{
		cout << "No weight filler parameter found! Initializing weights with xavier distribution." << endl;
		initialize_weights(_weights);
	}
	else
	{
		initialize_weights(_weights, _inner_product_parameter.weight_filler());
	}

	//bias setup
	if(_inner_product_parameter.has_bias_filler())
	{
		_bias_weights.resize(_inner_product_parameter.num_output(), 0);
		_bias_weights_delta.resize(_inner_product_parameter.num_output(), 0);
		double val = _inner_product_parameter.bias_filler().has_value() ? 
			_inner_product_parameter.bias_filler().value() : 0;
		_bias_value = val;
		initialize_weights(_bias_weights, _inner_product_parameter.bias_filler());
	}

	//initialize top

	//get batch size. There is only one bottom so this method of doing it is ok.  
	size_t batch_size = get_bottom_vector().front()->get_batch_size();

	vector<string>& tops = get_tops();

	if(tops.size() > 1) 
		cout << "WARNING: Layer " << _layer_parameter.name() << " has too many tops!" << endl;

	_output_dims = skepu2::Vector<size_t>{1, 1, _inner_product_parameter.num_output(), batch_size};
	_nr_neurons = _inner_product_parameter.num_output();
	_top = setup_top(data, mapped_data, tops.front(), _output_dims);
	cout << "Number of neurons in layer " << _weights.total_rows() << endl;
#ifdef DEBUG
	save_state("beginning");
#endif
}

FullyConnectedLayer::~FullyConnectedLayer()
{
#ifdef DEBUG
    cout << "~fully connected layer() " << this->get_name() << endl;
    cout << "bottom" << endl;
    _bottom.reset();
    cout << "delta top" << endl;
    _delta_top.reset();
    cout << "top" << endl;
    _top.reset();
    cout << "delta bottom" << endl;
    _delta_bottom.reset();
    cout << "weights" << endl;
    _weights.flush();
    cout << "weights delta" << endl;
    _weights_delta.flush();
    cout << "bias weights " << _bias_weights.getAddress() << endl;
    _bias_weights.flush();
    
    cout << "bias weights delta " << _bias_weights_delta.getAddress() << endl;
    _bias_weights_delta.flush();
    cout << "done for" << _name << endl;
#endif
}

/*
 * setup delta and some variables
 */
void FullyConnectedLayer::setup(Data& data, Mapped_data& mapped_data,
		shared_ptr<Layer> prev) 
{ 
	vector<string> deltas(get_deltas()); 
	_previous_layer = prev;

	//get batch size. There is only one bottom so this method of doing it is ok.  
	//size_t batch_size = get_bottom_vector().front()->get_batch_size();

	//setup delta vectors 
	const string& bottom_delta = deltas.back(); 

	//send to bottom layer 
	const string& top_delta = deltas.front(); 

	//receive from top layer 
	//size_t top_layer_neurons = next->get_weights()->total_rows(); 
	skepu2::Vector<size_t> delta_dims = _output_dims;
	//delta_dims[3] = 1; //batch is one in backward prop

	_delta_top = setup_delta_receiver(data, mapped_data, top_delta, 
			delta_dims); 
	
	_delta_bottom = setup_delta_sender(mapped_data, bottom_delta); 

#if DEBUG>1
	assert(_previous_layer->get_nr_neurons() > 0 );
	//cout << "weights " << _weights << endl;
	//cout << "bias weights " << _bias_weights << endl;
	cout << "bias values " << _bias_value << endl;
	save_weights(_weights);
#endif
}


/*
 * forward function definition
 */
void FullyConnectedLayer::forward(bool display) 
{ 
#ifdef DEBUG
	cout << "Activation function: " << Parser::activation_f_to_string(_activation_type) << endl;
#endif

	/*
	 * temp hard coded for the deep fully connected layers 
	 */
	double A = 1, B = 1;
	/*if(_activation_type == TANH_ACTIVATION)
	{
		A = 1.7159;
		B = 0.6666;
	}*/
	_skepu_f.forward(*_bottom, _weights, *_top, _activation_type, _inner_product_parameter.has_bias_filler(), 
			_bias_value, _bias_weights, A, B); 
}

/******************************************************************
 * backward function definition
 */
 void FullyConnectedLayer::backward() 
{
	//calculate errors 
	_skepu_f.error_calculation(*_delta_bottom, 
			*_delta_top,
			_weights,
			_previous_layer->get_activation_type(), 
			*_bottom);

#ifdef DEBUG
	//cout << *_delta_bottom << endl;
	double r = _skepu_f.average(_delta_bottom->_data) ;
	cout << "average out delta " << r << endl;
	assert( r > -100 && r < 100 );
#endif
}

/******************************************************************
 *
 * weight update
 *
 */
void FullyConnectedLayer::update_weights(Solver& solver) 
{
	//the delta recieved
	DataPackage& delta_top = *get_delta_vector().front();

#if DEBUG>1
	//cout << "weights before update " << endl << _weights << endl;
	auto copy_w(_weights);
#endif

	solver.update_weights(_weights, 
			_weights_delta,
			delta_top,
			*_bottom,
			_inner_product_parameter.has_bias_filler(),
			_bias_value,
			_bias_weights,
			_bias_weights_delta); 

#if DEBUG>1
	//cout << "weights after update " << endl << _weights << endl;
	if(copy_w == _weights) 
	{
		throw runtime_error("Weights have not been updated!");
	}
#endif

/*#ifdef DEBUG
	cout << "weights ";
	for(auto& i: _weights)
	{
		cout << i << ' ';
	}
	cout << endl;
#endif*/
}

void FullyConnectedLayer::save_state(string state) 
{ 
#ifdef DEBUG
	string s = "_" + state;
	save_weights(_weights, &s);
	s =  "bias_weights_" + state;
	save_weights(_bias_weights, &s);
#endif
}

void FullyConnectedLayer::load_state() { load_weights(_weights); }

void FullyConnectedLayer::test_preparation() { save_state("finished_training"); }
