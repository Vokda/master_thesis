#include "parser.hpp"
#include <algorithm>
#include <sstream>

Parser::Parser()
{
	//map data types to string
	_type_strings[DataType::INT] = "SKEPU::VECTOR<INT>";
	_type_strings[DataType::MONOCHROME_IMAGE] = "SKEPU::MATRIX<FLOAT>";
	_type_strings[DataType::COLOURED_IMAGE] =  "SKEPU::MATRIX<RGB>";
	_type_strings[DataType::DOUBLE] = "SKEPU::VECTOR<DOUBLE>";
	_type_strings[DataType::NOT_SUPPORTED] = "UNKNOWN TYPE";

	//map string to layer type
	_layer_map["input"] = INPUT;
	_layer_map["data"] = INPUT;
	_layer_map["innerproduct"] = FULLY_CONNECTED;
	_layer_map["softmaxwithloss"] = SOFT_MAX; 
	_layer_map["euclidiean"] = EUCLIDIEAN;
	_layer_map["pooling"] = POOL;
	_layer_map["sigmoid"] = SIGMOID;
	_layer_map["convolutional"] = CONVOLUTIONAL;
	_layer_map["convolution"] = CONVOLUTIONAL;
	_layer_map["relu"] = RELU;
	_layer_map["tanh"] = TANH;

	_string_types["mnist"] =		DataSet::MNIST;
	_string_types["cifar"] =		DataSet::CIFAR;
	//_string_types["image_net"] =	DataSet::IMAGE_NET;
	_string_types["test"] =			DataSet::TEST;
	_string_types["label"] =		DataSet::LABEL;
	_string_types["labels"] =		DataSet::LABEL;
}

DataSet Parser::get_data_set(string name) const
{
	if(name.empty())
	{
		cout << "Parser: No data set given!" << endl;
		return DataSet::UNKNOWN;
	}
	transform(name.begin(), name.end(), name.begin(), ::tolower);
	try
	{
		//if test set contains test in name return TEST dataset
		if(name.find("test") != string::npos) 
			return DataSet::TEST;
		else
			return _string_types.at(name);
	}
	catch(const out_of_range&  oor)
	{
		cout << "Warning: Parser: unknown data set: " << name << endl;
		return DataSet::UNKNOWN;
	}
}

string Parser::type_to_string(DataType d) const
{
	try
	{
		return _type_strings.at(d);
	}
	catch(const out_of_range& oor)
	{
		cout << "Warning: Parser: unknown data type" << endl;
		return "Unknown";
	}
}

DataType Parser::get_data_type(DataSet data) const
{
	switch(data)
	{
		case DataSet::MNIST:
		case DataSet::TEST:
			return DataType::MONOCHROME_IMAGE;
			break;
		//case DataSet::IMAGE_NET:
		case DataSet::CIFAR:
			return DataType::COLOURED_IMAGE;
			break;
		case DataSet::LABEL: //TODO: all labels might not be int
			return DataType::DOUBLE;
			break;
		default:
			throw runtime_error("Unknown data set!");
			break;
	}
}

LayerType Parser::get_layer_type(string type) const
{
	transform(type.begin(), type.end(), type.begin(), ::tolower);
	try
	{
		return _layer_map.at(type);
	}
	catch(const out_of_range& oor)
	{
		stringstream ss;
		ss << "Parser: Type of layer not recognized: \"" << type << "\"" << endl;
		throw runtime_error(ss.str());
		return NOT_A_TYPE;
	}
}

string Parser::activation_f_to_string(ActivationType A)
{
	switch(A)
	{
		case NO_ACTIVATION:
			return "identity";
		case SIGMOID_ACTIVATION:
			return "sigmoid";
		case RELU_ACTIVATION:
			return "ReLU";
		case TANH_ACTIVATION:
			return "TanH";
		default:
			throw runtime_error("Unknown activation function!");
			return "";
	}
}
