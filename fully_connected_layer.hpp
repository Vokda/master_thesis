#ifndef LAYER_FULLY_CONNECTED_H
#define LAYER_FULLY_CONNECTED_H
#include "layer.hpp"
#include "skepu_fully_connected.hpp"
//#include "solvers.hpp"

class Solver;
class MetaLayer;

using namespace std;

class FullyConnectedLayer: public Layer
{
	public:
		FullyConnectedLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				Data& data, Mapped_data&);
		~FullyConnectedLayer();

		void setup(Data& data, Mapped_data&, shared_ptr<Layer> next);

		void forward(bool display);
		void backward();
		void update_weights(Solver& s);
		void test_preparation();

		//weights
		const Weights* get_weights() const { return &_weights; }
		Weights* get_weights() { return &_weights; }

		void save_state(string state="");
		void load_state();

	private:

		InnerProductParameter _inner_product_parameter;

		/* weights
		 * rows represent the individual neurons
		 * columns contains the actual weights for a neuron
		 */
		Weights _weights;
		//weight change from preivous iteration, used with momentum in weight update;
		Weights _weights_delta;

		//skepu functionallity 
		SkePU_FullyConnected _skepu_f;

		skepu2::Vector<size_t> _output_dims;

		//bias related members
		skepu2::Vector<double> _bias_weights;
		//weight change from preivous iteration, used with momentum in weight update;
		skepu2::Vector<double> _bias_weights_delta;
		double _bias_value;

		//input 
		shared_ptr<DataPackage> _bottom;
		shared_ptr<DataPackage> _delta_top;
		
		//output
		shared_ptr<DataPackage> _top;
		shared_ptr<DataPackage> _delta_bottom;
};
#endif
