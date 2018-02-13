#include "layer_factory.hpp"
#include <algorithm>
#include <string>
#include <sstream>
#include "layers.hpp"
#include "topological_sort.hpp"
#include <stdexcept>
#include "layers.hpp"


LayerFactory::LayerFactory()
{
}

Layer* LayerFactory::make_layer(const node& n, const shared_ptr<skepu2::BackendSpec> spec,
				vector<shared_ptr<DataPackage>>& data_ref, Mapped_data& mapped_data) const
{
	vector<DataType> types;
	for(auto bottom: n.get_receive_from())
	{
		types.push_back(bottom->type);
	}

	//only input layer should have no bottom.
	//input layer should probably have been done in created_templated layer instead
	if(types.empty())
	{
		DataType type;
		const LayerParameter& node_layer_param = n.get_layer_parameter();
		//make sure the input layer actually has input to read
		if(node_layer_param.has_data_param())
		{
			const string& source = n.get_layer_parameter().data_param().source();
			type = _parser.get_data_type(_parser.get_data_set(source));
		}
		else
		{
			stringstream ss;
			ss << n.get_name() << " does not have a source member." << endl;
			throw runtime_error(ss.str());
		}

		return new InputLayer(n, spec, data_ref, mapped_data);
			/*
		switch(type)
		{
			//case vector<DataPackage>Type::COLOURED_IMAGE:
			default:
				cout << "Layer factory: Type of data not supported or recognized: " 
					<< _parser.type_to_string(type)  << endl;
				return nullptr;
				break;
		}
		*/
	}
	else
		return create_templated_layer(n, spec, types, data_ref, mapped_data);
}


Layer* LayerFactory::create_templated_layer(const node& n, const shared_ptr<skepu2::BackendSpec> spec, 
		vector<DataType> types,  vector<shared_ptr<DataPackage>>& data_ref, Mapped_data& mapped_data) const
{
	/*
#ifdef DEBUG
	cout << __FUNCTION__ << endl;
	for(auto& i: mapped_data)
	{
		cout << i.first << endl;
		shared_ptr<DataPackage> find = nullptr;
		for(auto& j: data_ref)
		{
			cout << "is " << &j << " == " << i.second << endl;
			if(j == i.second)
			{
				cout << "yes!" << endl;
				find = j;
				break;
			}
		}
		cout << "find " << find << endl;
		cout << "i.second " << i.second << endl;
		cout << *i.second << endl;
		assert(i.second == find);
	}
#endif*/

	LayerType lp = n.get_layer_type();
	switch(lp)
	{
		case CONVOLUTIONAL:
			return new ConvolutionalLayer(n, spec, data_ref, mapped_data);
		case POOL:
			return new PoolingLayer(n, spec, data_ref, mapped_data);
		case FULLY_CONNECTED:
			return new FullyConnectedLayer(n, spec, data_ref, mapped_data);
		case SOFT_MAX:
			return new SoftMaxLayer(n, spec, data_ref, mapped_data);
		case NOT_A_TYPE:
			cout << "Layer factory: layer of type " << lp 
				<< " could not be made with one argument." << endl;
		default:
			return nullptr;
	}
}
