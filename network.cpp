#include "network.hpp"
#include <iostream>
#include <sstream>
#include "layer.hpp"
#include "layer_factory.hpp"
#include <skepu2.hpp>
#include "proto_io.hpp"
#include <milli.hpp>

static Parser parser;
static Proto_io pio;

Network::Network(const SolverParameter& sp, const std::shared_ptr<skepu2::BackendSpec> spec):
	_solver_parameter(sp)
{
	//set network parameter;
	try
	{
		pio.read_prototxt(sp.net(), &_net_parameter);
	}
	catch(const exception& e)
	{
		throw runtime_error(e.what());
	}
	cout << "--------- Constructing network " <<_net_parameter.name() << " ---------" << endl;
	cout << "Number of layers to create: " << _net_parameter.layer_size() << endl;

	cout << "Creating nodes (meta layers) to be sorted topologically." << endl;
	//create meta layers
	create_meta_layers();

	//sort layers so that data is setup in correct order
	_tsort();

	cout << "Creating actual layers from nodes created earlier." << endl;
	//create layers from the meta layers created earlier
	create_layers(spec);

	cout << "--- Layer creation completed. Setting up layers. ---" << endl;

	shared_ptr<Layer> previous_layer;
	//setup data, initialize weights and such
	for(size_t i = 0; i < _layers.size(); ++i)
	{
		if(i > 0)
			previous_layer = _layers[i-1];
		string layer_name = _layers[i]->get_name();
		cout << "--- Setting up layer " << layer_name << " ---" << endl;
		_layers[i]->setup(_data, _mapped_data, previous_layer);
		cout <<  "--- Setup for layer " << layer_name << " completed. ---" << endl << endl;
		
	}

#if DEBUG>1
	cout << "--DEBUG OUTPUT--" << endl;
	cout << "Data container (size " << _data.size() << "):" << endl;
	for(auto i: _data)
	{
		cout << *i << endl;
	}
#endif
	cout << "--------- Network constrution completed ---------" << endl;
}

void Network::create_meta_layers()
{
	vector<const LayerParameter*> layer_parameters;
	for(int i = 0; i < _net_parameter.layer_size(); ++i)
	{
		layer_parameters.push_back(&_net_parameter.layer(i));
	}
	_tsort.setup(layer_parameters, _solver_parameter);
}
	

void Network::create_layers(shared_ptr<skepu2::BackendSpec> spec)
{
	auto& node_graph = _tsort.get_graph();
	for(size_t i = 0; i < node_graph.size(); ++i)
	{
		auto& meta_layer = node_graph[i];
		cout << "---------- Layer to be constructed: " << meta_layer.get_layer_parameter().type() << " ----------"<< endl;
		cout << "Name of layer: " << meta_layer.get_layer_parameter().name() << endl;

		//make layer
		//Layer* layer = _layer_factory.make_layer(meta_layer, spec, _data, _mapped_data); 
		shared_ptr<Layer> layer(_layer_factory.make_layer(meta_layer, spec, _data, _mapped_data));

		if(layer != nullptr)
		{
			_layers.push_back(layer);	
			//_layers.push_back(layer);	
		}
		else
		{
			throw runtime_error("Network: network could not be made."); 
		}
	}
	cout << "Number of layers constructed: " << node_graph.size() << endl;;
	cout << "NUmber of hidden layers: " << node_graph.size() -2;
}

void Network::make_call_order()
{
	//_tsort(_layers);
}

//////////////////// FORWARD //////////////////////////////////
bool Network::forward(bool display)
{
    _display = display;
	//actual forward function
	for(layer_itr layer = _layers.begin(); layer != _layers.end(); ++layer)
	{
#ifdef DEBUG
		cout << "---------- Layer: " << (*layer)->get_type() << " - "
			<< (*layer)->get_name() << ": forward() --------" << endl; 
#endif
		_start_time = milli::GetSeconds();

		(*layer)->forward(_display);

		_end_time = milli::GetSeconds();
		_time = _end_time - _start_time;
		if(_display)
			cout << "forward time " << _time << endl;

	}
	//if having run through all of the test data abort run
	return _layers.front()->is_at_last_batch();
}

//////////////////// BACKWARD //////////////////////////////////
bool Network::backward()
{
/*#ifdef DEBUG
	cout << "layers " << endl;
	for(auto i: _layers)
		cout << i->get_name() << endl;
	cout << "layer size " << _layers.size() << endl;
#endif*/
	//rend()-2 because the input layer and the layer above does not need to call their backward()
    //rbegin()+1 because the output layer performs the backpropagation in the forward function
	for(auto layer = _layers.rbegin(); layer != _layers.rend()-2; ++layer)
	{
		//const Layer* layer = (*i);
#ifdef DEBUG
		cout << "----------- Layer: " << (*layer)->get_type() << " - "<< (*layer)->get_name() << ": backward() -------------" << endl;
#endif

		_start_time = milli::GetSeconds();

		(*layer)->backward();

		_end_time = milli::GetSeconds();
		_time = _end_time - _start_time;

		if(_display)
			cout << "backward time " << _time << endl;
	}
	return _layers.front()->is_at_last_batch();
}

///////////////////////// UPDATE ////////////////////////
void Network::update_weights(Solver& s)
{
	for(auto layer: _layers)
	{
#ifdef DEBUG
		cout << "------------- Layer: " << layer->get_type() << " - "<< layer->get_name() << ": update_weights() ---------------" << endl;
#endif
		_start_time = milli::GetSeconds();

		layer->update_weights(s);

		_end_time = milli::GetSeconds();
		_time = _end_time - _start_time;
		if(_display)
			cout << "update time " << _time << endl;
	}
}


void Network::test_preparation()
{
	cout << "----------- Test Preparation. ---------------" << endl;
	for(auto& layer: _layers)
	{
		cout << "Layer " << layer->get_name() << " doing test preparation." << endl;
		layer->test_preparation();
	}
    cout << "--- Test Preparation Completed! ---" << endl;
}

void Network::post_test()
{
	cout << "--- Post Test ---" << endl;
	for(auto& layer: _layers)
	{
		cout << "Layer " << layer->get_name() << " doing post test." << endl;
		layer->post_test();
	}
    cout << "--- Post Test Completed! ---" << endl;
}
