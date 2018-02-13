#include "layer.hpp"
#include <skepu2.hpp>
#include "meta_layer.hpp"
#include <sstream>

Layer::Layer(const MetaLayer& ml, shared_ptr<skepu2::BackendSpec> spec):
	_solver_parameter(ml.get_solver_parameter()),
	_layer_parameter(ml.get_layer_parameter()),
	_activation_type(ml.get_activation_type()),
	_backend_specification(spec),
	_previous_layer(nullptr),
	_nr_neurons(-1), //so that it has to be set.
	_randomizer(_solver_parameter.random_seed()),
    _last_batch(false), _testing(false)
{
	if(ml.get_layer_type() == FULLY_CONNECTED or
			ml.get_layer_type() == CONVOLUTIONAL)
		cout << "Activation function set: " << Parser::activation_f_to_string(_activation_type) << endl;
	_name = ml.get_layer_parameter().name();
	_type = ml.get_layer_parameter().type();
	assert(!_name.empty());
	assert(!_type.empty());
	
	//bottom names
	for(int i = 0; i < _layer_parameter.bottom_size(); ++i)
	{
		string name = _layer_parameter.bottom(i);
		_bottom_names.push_back(name);
	}

	//top names
	for(int i = 0; i < _layer_parameter.top_size(); ++i)
	{
		string name = _layer_parameter.top(i);
		_top_names.push_back(name);
	}

	//deltas
	for(const MetaLayer::meta_data& md: ml.get_deltas())
	{
		_delta_names.push_back(md.name);
	}

#ifdef DEBUG
	/*cout << __FUNCTION__ << endl;
	cout << "delta_names:" << endl;
	for(auto& i: _delta_names)
	{
		cout << i << endl;
	}
	cout << "meta_layer vector send_to:" << endl;
	for(auto& i: ml.get_send_to())
	{
		cout << i->name << endl;
	}*/
#endif
	//for(const MetaLayer::meta_data* md: ml.get_send_to())
	for(auto* md: ml.get_send_to())
	{
		_delta_names.push_back(md->name);
	}
}

Layer::~Layer()
{
#ifdef DEBUG
    cout << "running destructor for base for  " << _name << endl;
    cout << "top data" <<endl;
    for(auto& i: _top_data)
    {
        cout << "name " << i->_name << ", address " << &*i << endl;
        i.reset();
    }
    cout << "bottom data" <<endl;
    for(auto& i: _bottom_data)
    {
        i.reset();
    }
    cout << "delta_ data" << endl;
    for(auto& i: _delta_data)
    {
        i.reset();
    }
    cout << "done with base for " << _name << endl;
#endif
}

vector<string>& Layer::get_tops()
{
	return _top_names;
}

vector<string>& Layer::get_bottoms()
{
	return _bottom_names;
}

vector<string>& Layer::get_deltas()
{
	return _delta_names;
}

vector<string> Layer::get_tops_and_bottoms()
{
	vector<string> names = {get_bottoms()};
	vector<string> names_tops = {get_tops()};
	names.insert(names.end(), names_tops.begin(), names_tops.end());
	return names;
}

/////////////////SETUP FUNCTIONS AND SPECIALIZATIONS/////////////////

shared_ptr<DataPackage> Layer::setup_top(
		Data& data, 
		Mapped_data& mapped_data,
		const string& data_name, skepu2::Vector<size_t>& dims)
{
	//add to the network storage. Network class will dealocate the memory
	//data.push_back(new DataPackage{dims, data_name}); 
	data.emplace_back(new DataPackage(dims, data_name));
	shared_ptr<DataPackage> dp(data.back());

	//add pointer to layer top data (might be unnecessary) TODO
	_top_data.push_back(dp);

	//map data pointer to string, this is for the next layer(s) to be able to locate the data
	mapped_data[data_name] = dp;

	cout << "Data mapped to \"" << data_name <<"\"" << endl;
/*#ifdef DEBUG
	cout << __FUNCTION__ << '&' << endl;
	cout <<"mapped to " << mapped_data[data_name] << endl;
	cout << "data addresss " << &data.back() << endl;
	cout << *mapped_data[data_name] << endl;
#endif*/
	return dp;
}

shared_ptr<DataPackage> Layer::setup_top(Data& data, 
		Mapped_data& mapped_data,
		const string& data_name, skepu2::Vector<size_t>&& dims)
{
	//add to the network storage. Network class will dealocate the memory
	data.emplace_back(new DataPackage(dims, data_name));
	//data.push_back(DataPackage{dims, data_name}); 
	shared_ptr<DataPackage> dp(data.back());

	//add pointer to layer top data (might be unnecessary) TODO
	_top_data.push_back(dp);

	//map data pointer to string, this is for the next layer(s) to be able to locate the data
	mapped_data[data_name] = dp;

	cout << "Data mapped to \"" << data_name <<"\"" << endl;
/*#ifdef DEBUG
	cout << __FUNCTION__ << "&&" << endl;
	cout << mapped_data[data_name] << endl;
	cout << *mapped_data[data_name] << endl;
#endif*/
	return dp;
}

shared_ptr<DataPackage> Layer::setup_bottom(Mapped_data& mapped_data, const string& bottom)
{
	cout << "Trying to find \"" << bottom << "\"."<< endl;
	auto key = mapped_data.find(bottom);
/*
#ifdef DEBUG
	cout << __FUNCTION__ << endl;
	for(auto& i: mapped_data)
	{
		cout << i.first << endl;
		cout << i.second << endl;
		cout << *i.second << endl;
	}
#endif
	*/
	//if found, push the pointer to the bottom data 
	if(key != mapped_data.end())
	{
		cout << "\"" << bottom << "\" found! Storing pointer to data." << endl;
		_bottom_data.push_back(key->second);
/*#ifdef DEBUG
		cout << key->second << endl;
		cout << *key->second << endl;
#endif*/
		return key->second;
	}
	else
	{
		std::stringstream ss;
		ss << "Data package \"" << bottom << "\" not found!" << endl;
		throw(runtime_error(ss.str()));
		return nullptr;
	}
}

//similar to setup bottom, no allocation
shared_ptr<DataPackage> Layer::setup_delta_sender(Mapped_data& mapped_data, const string& delta)
{
	cout << "Trying to find \"" << delta << "\"."<< endl;
	auto key = mapped_data.find(delta);
	//if found, push pointer to the delta data
	if(key != mapped_data.end())
	{
		cout << "\"" << delta << "\" found! Storing pointer to data." << endl;
		//shared_ptr<DataPackage> dat = key->second;
		_delta_data.push_back(key->second);
		return key->second;
	}
	else
		cout << "Warning: " << delta << " not found!" << endl;
}

shared_ptr<DataPackage> Layer::setup_delta_receiver(Data& data, Mapped_data& mapped_data, 
		const string& data_name, skepu2::Vector<size_t>& dims)
{
	//allocate data
	//data.push_back(DataPackage{dims, data_name});
	data.emplace_back(new DataPackage(dims, data_name));
	shared_ptr<DataPackage> dpb(data.back());
	//add to network class's storage
	//add pointer to delta vector
	_delta_data.push_back(dpb);
	//map data pointer to string, this is for the sender layer(s) to be able to locate the data
	mapped_data[data_name] = dpb;
	cout << "Delta mapped to \"" << data_name <<"\"";
	//cout << "Batch size: " <<  << ". vector<DataPackage> size: " << delta_values << endl;
/*#ifdef DEBUG
	cout << *data.back() << endl;
#endif*/
	return dpb;
}
		
shared_ptr<DataPackage> Layer::setup_delta_receiver(Data& data, Mapped_data& mapped_data, 
		const string& data_name, skepu2::Vector<size_t>&& dims)
{
	//allocate data
	data.emplace_back(new DataPackage(dims, data_name));
	//data.push_back(DataPackage{dims, data_name});
	//DataPackage* dpb = &data.back();
	shared_ptr<DataPackage> dpb(data.back());
	//add to network class's storage
	//add pointer to delta vector
	_delta_data.push_back(dpb);
	//map data pointer to string, this is for the sender layer(s) to be able to locate the data
	mapped_data[data_name] = dpb;
	cout << "Delta mapped to \"" << data_name <<"\"";
	//cout << "Batch size: " <<  << ". vector<DataPackage> size: " << delta_values << endl;
/*#ifdef DEBUG
	cout << *data.back() << endl;
#endif*/
	return dpb;
}
///////////////////////////WEIGHT SPECIFIC FUNCTIONS//////////////////////

void Layer::initialize_weights(Weights& w)
{
	_randomizer.xavier(w, 0.0, 0.05);
}

void Layer::initialize_weights(DataPackage& w)
{
	_randomizer.xavier(w._data, 0.0, 0.05);
}


void Layer::initialize_weights(Weights& w, const FillerParameter& fp)
{
	string type = fp.type();
	transform(type.begin(), type.end(), type.begin(), ::tolower);
	if(type == "xavier")
	{
		cout << "Initalizing weights with xavier distribution." << endl;
		double min = fp.has_min() ? fp.min() : -0.5;
		double max = fp.has_max() ? fp.max() : 0.5;
		cout << "min " << min << endl;
		cout << "max " << max << endl;
		_randomizer.xavier(w, min, max);
	}
	else if(type == "uniform")
	{
		cout << "Initalizing weights with uniform distribution." << endl;
		double min = fp.has_min() ? fp.min() : -0.05;
		double max = fp.has_max() ? fp.max() : 0.05;
		cout << "min " << min << endl;
		cout << "max " << max << endl;
		_randomizer.uniform(w, min, max);
	}
	else if(type == "gaussian")
	{
		cout << "Initalizing weights with gaussian distribution." << endl;
		double mean = fp.has_mean() ? fp.mean() : 0;
		double std = fp.has_std() ? fp.std() : 1;
		cout << "mean " << mean << endl;
		cout << "std " << std << endl;
		_randomizer.gaussian(w, mean, std);
	}
	else 
	{
		string s("Layer: weight init selected not supported: " + type);
		throw runtime_error(s);
	}
}

void Layer::initialize_weights(DataPackage& w, const FillerParameter& fp)
{
	initialize_weights(w._data, fp);
}

void Layer::initialize_weights(skepu2::Vector<double>& v, const FillerParameter& fp)
{

	string type = fp.type();
	transform(type.begin(), type.end(), type.begin(), ::tolower);

	if(type == "xavier")
	{
		cout << "Initalizing weights with xavier distribution." << endl;
		double min = fp.has_min() ? fp.min() : -0.5;
		double max = fp.has_max() ? fp.max() : 0.5;
		cout << "min " << min << endl;
		cout << "max " << max << endl;
		_randomizer.xavier(v, min, max);
	}
	else if(type == "uniform")
	{
		cout << "Initalizing weights with uniform distribution." << endl;
		double min = fp.has_min() ? fp.min() : -0.05;
		double max = fp.has_max() ? fp.max() : 0.05;
		cout << "min " << min << endl;
		cout << "max " << max << endl;
		_randomizer.uniform(v, min, max);
	}


	else if(type == "gaussian")
	{
		cout << "Initalizing weights with gaussian distribution." << endl;
		double mean = fp.has_mean() ? fp.mean() : 0;
		double std = fp.has_std() ? fp.std() : 1;
		_randomizer.gaussian(v, mean, std);
	}
	else if(type == "constant")
	{
		cout << "Initalizing weights with constant distribution." << endl;
		_randomizer.constant(v, fp.value());
	}
	else 
	{
		string s("Layer: weight init selected not supported: " + type);
		throw runtime_error(s);
	}
}




void Layer::save_weights(Weights& w) const
{
	//string file_name =  + "train_delta.dat";
	stringstream ss;
	ss << _solver_parameter.snapshot_prefix() << get_name() << ".weights";
	std::ofstream save_file (ss.str(), ios::trunc);
	//write data
	for(size_t i = 0; i < w.size(); ++i)
	{
		save_file << w[i] <<' ';
	}
	save_file.close();

	cout << get_name() << "\'s weights saved as " << ss.str() << endl;
}

void Layer::save_weights(DataPackage& data_package) const
{
	stringstream ss;
	ss << _solver_parameter.snapshot_prefix() << get_name() << ".weights";
	std::ofstream save_file (ss.str(), ios::trunc);
	//write data
	for(size_t i = 0; i < data_package._data.size(); ++i)
	{
		save_file << data_package._data[i] << ' ';
	}
	save_file.close();

	cout << get_name() << "\'s weights saved as " << ss.str() << endl;
}

void Layer::load_weights(Weights& w)
{
	stringstream ss;
	ss << _solver_parameter.snapshot_prefix() + '/' << get_name() << ".weights";
	std::ifstream file(ss.str());
	if(file.is_open())
	{
		for(size_t i = 0; i < w.size(); ++i)
		{
			file >> w[i];
		}
	}
	else
		cout << "File " << ss.str() << " could not be opened!" << endl;
}

void Layer::load_weights(DataPackage& w)
{
	stringstream ss;
	ss << _solver_parameter.snapshot_prefix()  + '/'<< get_name() << ".weights";
	std::ifstream file(ss.str());
	if(file.is_open())
	{
		for(size_t i = 0; i < w._data.size(); ++i)
		{
			file >> w[i];
		}
	}
	else
		cout << "File " << ss.str() << " could not be opened!" << endl;
}
