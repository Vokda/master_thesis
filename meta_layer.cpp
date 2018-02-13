#include "meta_layer.hpp"
#include "proto_io.hpp"

#ifdef DEBUG
#include <iostream>
using namespace std;
#endif

static Parser parser;

MetaLayer::MetaLayer(const SolverParameter& sp, const LayerParameter& lp):
	layer_parameter(lp), solver_parameter(sp), discovery_time(0), finishing_time(0), color(Color::WHITE),
	layer_type(parser.get_layer_type(lp.type())),
	_activation_type(NO_ACTIVATION)
{
	//add bottom layers to node
	for(int i = 0; i < layer_parameter.bottom_size(); ++i)
	{
		string data_name = layer_parameter.bottom(i);
		meta_data md;
		md.name = data_name;
		bottoms.emplace_back(md);
	}

	//add top layers to node
	for(int i = 0; i < layer_parameter.top_size(); ++i)
	{
		string data_name = layer_parameter.top(i);
		meta_data md;
		md.name = data_name;
		tops.emplace_back(md);
	}

	//set activation type if it has one
	if(is_activation_layer())
	{
		set_activation_type(layer_type);
	}
	

#ifdef DEBUG
	cout << "Layer " << get_name() << endl;
	cout << "Type: " << lp.type() << endl;
#endif
}

MetaLayer::MetaLayer(const MetaLayer& m):
	layer_parameter(m.layer_parameter),solver_parameter(m.solver_parameter)
{
	if(this != &m)
	{
		neighbors = m.neighbors;
		receive_from = m.receive_from;
		tops = m.tops;
		bottoms = m.bottoms;
		deltas = m.deltas;
		send_to = m.send_to;
		_activation_type = m._activation_type;
		layer_type = m.layer_type;
	}
}
MetaLayer& MetaLayer::operator=(MetaLayer&& m)
{
	if(this != &m)
	{
		layer_parameter = m.layer_parameter;
		//solver_parameter = m.solver_parameter;
		neighbors = m.neighbors;
		receive_from = m.receive_from;
		tops = m.tops;
		bottoms = m.bottoms;
		deltas = m.deltas;
		send_to = m.send_to;
		_activation_type = m._activation_type;
		layer_type = m.layer_type;
		return *this;
	}
	else
		return *this;
}

MetaLayer::meta_data* MetaLayer::add_delta_data(string s)
{
	meta_data md;
	md.name = s;
	md.type = DataType::DOUBLE;
	deltas.emplace_back(md);
	return &deltas.back();
}

bool MetaLayer::is_activation_layer() const
{
	switch(get_layer_type())
	{
		case LayerType::RELU:
		case LayerType::SIGMOID:
		case LayerType::TANH:
			return true;
		default:
			return false;
	}
	return false;
}

void MetaLayer::set_activation_type(LayerType t)
{
	switch(t)
	{
		case LayerType::RELU:
			_activation_type = RELU_ACTIVATION;
			break;
		case LayerType::SIGMOID:
			_activation_type = SIGMOID_ACTIVATION;
			break;
		case LayerType::TANH:
			_activation_type = TANH_ACTIVATION;
			break;
		default:
			_activation_type = NO_ACTIVATION;
			break;
	}
}
