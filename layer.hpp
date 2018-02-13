#ifndef LAYER_H
#define LAYER_H
#include "proto/caffe.pb.h"
#include "weight_randomizer.hpp"
#include <memory>
#include "data_structure.hpp"
#include <type_traits>
#include <typeinfo>

class MetaLayer;
class Solver;

using namespace caffe;
using namespace std;

class Layer
{
	public:
		Layer(const MetaLayer& ml, shared_ptr<skepu2::BackendSpec> spec);

		virtual ~Layer();

		//! forward function of the layer
		virtual void forward(bool display) = 0;

		//!will setup the layer appropriately.
		virtual void setup(Data& data_ref, Mapped_data&, shared_ptr<Layer> previous) = 0;

		/**
		 * Backward funciton of the layer
		 * will calculate delta vectors (errors) of the network
		 */
		virtual void backward() = 0;

		//! Will update the weights of the layer. (only layers with weights will be affected)
		virtual void update_weights(Solver&) = 0;

		const string& get_name() const
		{
			//return _layer_parameter.name();
			return _name;
		}

		const string& get_type() const
		{
			//if(!_layer_parameter.has_type())
				//throw runtime_error("has no type");
			//return _layer_parameter.type();
			return _type;
		}
		
		const LayerParameter& get_layer_parameter() const
		{
			return _layer_parameter;
		}

		const skepu2::BackendSpec* get_backend() const
		{
			return _backend_specification.get();
		}

		//!This funciton will save the state of the layer
		virtual void save_state()
		{
#ifndef REMOVE_OUTPUT
			cout << get_name() << " does not have a state to save. " << endl;
#endif
		}

		//!This funciton will save the state of the layer
		virtual void load_state()
		{
#ifndef REMOVE_OUTPUT
			cout << get_name() << " does not have a state to load. " << endl;
#endif
		}

		virtual void test_preparation()
		{
			_testing = true;
		}

		virtual void post_test()
		{
		}

		bool is_at_last_batch()
		{ 
			return _last_batch;
		}

		//! get activation type
		ActivationType get_activation_type() const { return _activation_type; }


	protected:
		//typedefs
		typedef map<string, DataPackage*> data_map;
		
		/*
		 * VARIABLES
		 */

		SolverParameter _solver_parameter;
		LayerParameter _layer_parameter;
		//layer parameter related vars
		string _name;
		string _type;

		size_t _nr_neurons;

        shared_ptr<Layer> _previous_layer;

		//true if testing
		bool _testing;

		//true if it is the last batch
		bool _last_batch;

		//!if the layer is merged with an activation layer it will be set to something different.
		ActivationType _activation_type{ActivationType::NO_ACTIVATION};

		/*
		 * SETUP DECLARATIONS
		 */


		/**
		 * This function can be used to setup the bottom data (input to layer) properly 
		 * so that the layer can use later.
		 * This is the only function where the map should be used to avoid performance obstacles.
		 */
		shared_ptr<DataPackage> setup_bottom(Mapped_data& mapped_data, const string& data_name);
		
		/**
		 * This function setups the top data (output from layer) properly 
		 * so that the next layer can find it.
		 */
		shared_ptr<DataPackage> setup_top(
				Data& data, 
				Mapped_data& mapped_data, 
				const string& data_name, 
				skepu2::Vector<size_t>& dimensions);

		shared_ptr<DataPackage> setup_top(
				Data& data, 
				Mapped_data& mapped_data, 
				const string& data_name, 
				skepu2::Vector<size_t>&& dimensions);
		/*
		 * The following two functions is similar to previous setup functions but for delta values.
		 * The delta sender is to be used by the layers calculating the delta values to write them to this data
		 * The delta receiver function is for the layers receiving delta values. 
		 * A layer may need to run both functions.
		 */

		/**
		 * will function similar to setup bottom.
		 * no data allocation, only lookup is done here.
		 */
		 shared_ptr<DataPackage> setup_delta_sender(Mapped_data& mapped_data, const string& data_name);

		/**
		 * This function is similar to setup_top.
		 * allocates data and map it so that it can be found by other layers
		 */
		shared_ptr<DataPackage> setup_delta_receiver(Data& data, 
				Mapped_data& mapped_data, 
				const string& data_name, skepu2::Vector<size_t>& dims);

		shared_ptr<DataPackage> setup_delta_receiver(Data& data, 
				Mapped_data& mapped_data, 
				const string& data_name, skepu2::Vector<size_t>&& dims);

		/*
		 * WEIGHT FUNCTIONS
		 */

		/**
		 * initialize weights
		 * This assumes a fully connected network
		 */
		void initialize_weights(Weights& w);
		void initialize_weights(DataPackage& w);

		void initialize_weights(Weights& w, const FillerParameter& fp);
		void initialize_weights(DataPackage& w, const FillerParameter& fp);
		void initialize_weights(skepu2::Vector<double>& w, const FillerParameter& fp) ;

		double initialize_weight() { return _randomizer.random(-0.05, 0.05); }

		/**
		 * saves skepu matrix to disk.
		 */
		void save_weights(Weights& w) const;
		void save_weights(Weights& w, const string& name) const;
		void save_weights(DataPackage& w) const;

		template<typename T>
		void save_weights(T& w, const string* name=nullptr) const
		{
			stringstream ss;
			string n;
			if(name != nullptr)
				n = '_' + *name;
			else
				n = "";
			ss << _solver_parameter.snapshot_prefix() << get_name() << n << ".weights";
			std::ofstream save_file (ss.str(), ios::trunc);
			//write data
			for(size_t i = 0; i < w.size(); ++i)
			{
				save_file << w[i] << ' ';
			}
			save_file.close();

			cout << get_name() << "\'s weights saved as " << ss.str() << endl;
		}

		/**
		 * loads weights from file.
		 * the containers must be set to the correct size before calling this function.
		 */
		void load_weights(Weights& w);
		void load_weights(DataPackage& w);

		/*
		 * ACCESSORS
		 */
		



		/*
		 * GET (AND CAST) ACCESSORS FOR DATA
		 */

		//get data
		Data& get_bottom_vector() { return _bottom_data; }
		const Data& get_bottom_vector() const { return _bottom_data; }

		Data& get_top_vector() { return _top_data; }
		const Data& get_top_vector() const { return _top_data; }

		Data& get_delta_vector() { return _delta_data; }
		const Data& get_delta_vector() const { return _delta_data; }

		//return only element in data vectors
		shared_ptr<DataPackage> get_only_bottom() { return _bottom_data.back(); };

		shared_ptr<DataPackage> get_only_top() { return _top_data.back(); };
		

		//get tops and bottoms 
		vector<string>& get_bottoms();
		vector<string>& get_tops();
		vector<string> get_tops_and_bottoms();
		vector<string>& get_deltas();
	public:

		virtual Weights* get_weights() 
		{ 
			stringstream ss;
			ss << __PRETTY_FUNCTION__ << ": trying to get weights from object that has none!" << endl; 
			throw runtime_error(ss.str());
		}
		virtual const Weights* get_weights() const 
		{ 
			stringstream ss;
			ss  << __PRETTY_FUNCTION__ << ": trying to get weights from object that has none!" << endl;
			throw runtime_error(ss.str());
		}

		//! returns the number of neurons
		size_t get_nr_neurons() const { return _nr_neurons; };


	private:
		/*
		 * TODO 
		 * possibly should be separate for each layer, such as the input layer containing two top
		 * pointers: labels and images;
		 */
		vector<shared_ptr<DataPackage>> _bottom_data;
		vector<shared_ptr<DataPackage>> _top_data;
		vector<shared_ptr<DataPackage>> _delta_data;

		vector<string> _top_names;
		vector<string> _bottom_names;
		vector<string> _delta_names;


		//skepu specific data
		shared_ptr<skepu2::BackendSpec> _backend_specification;

		WeightRandomizer _randomizer;
};
//#include "layer.inl"
#endif
