#ifndef NETWORK_H
#define NETWORK_H
#include "layer_factory.hpp"
#include "topological_sort.hpp"
#include "data_structure.hpp"
#include <memory>
//#include "util.hpp"

class Layer;
class Solver;

namespace skepu2
{
	struct BackendSpec;
}

using namespace std;

class Network
{
	public:
		Network(const SolverParameter& sp, const std::shared_ptr<skepu2::BackendSpec> spec);

        /*
		~Network()
		{
		}*/

		/**
		 * Will run the layers forward functions in order.
		 * Returns true if last batch of data has been used.
		 */
		bool forward(bool display); //run the net forward

		/**
		 * Will run the layers backward function in reverse order. 
		 * Returns true if the last batch of data has been used.
		 */
		bool backward();

		/**
		 * Updates the weights of all the layers that have weights
		 */
		void update_weights(Solver& s);

		/**
		 * This function should be called before the test phase.
		 * This allows layer to perform any changes needed before the test phase. 
		 */
		void test_preparation();

		/**
		 * This function should be called after the test phase for any final operations post test
		 */
		void post_test();

		vector<shared_ptr<Layer>>& get_layers() {return _layers;}

	private:

		//creates meta layers for sorting and creation later
		void create_meta_layers();

		//creates layers with layer_factory.
		void create_layers(shared_ptr<skepu2::BackendSpec> spec);

		//This will topological sort the _layers container. 
		void make_call_order();


		//vector<TopologicalSort::node*> _temp_nodes;
		//used for storing layers. Are sorted in topological order by _tsort
		vector<shared_ptr<Layer>> _layers;
        typedef vector<shared_ptr<Layer>>::iterator layer_itr;

		LayerFactory _layer_factory;
		NetParameter _net_parameter;
		TopologicalSort _tsort;

		const shared_ptr<skepu2::BackendSpec> _backend_specification;

		/**
		 * The stored data used by the network.
		 * Both data for forward and backward passes are stored here.
		 * See topological_sort.hpp for more details.
		 *
		 * Each skepu vector is a batch of data use by two or more layers.
		 * E.g. Layer1 sends data to layer2 by using _data[0]. 
		 * _data[0] is the processed by the recieving layer.
		 */
		vector<shared_ptr<DataPackage>> _data;

		//only used for connecting layers to the correct data when layers are created.
		map<string, shared_ptr<DataPackage>> _mapped_data;

		const SolverParameter& _solver_parameter;

		//used for timers
		double _start_time; //when a layer starts function
		double _end_time; //when a layer ends function
		double _time; //time taken for layer to perform function

		/*
		 * This is used to display only when the user wants to.
		 * data is not sent to solver class to make things easier.
		 */
		bool _display{false}; 
};
#endif
