#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H
#include "layer.hpp"
#include "skepu_convolutional.hpp"

class Solver;
class MetaLayer;

using namespace std;

class ConvolutionalLayer: public Layer
{
	public:
		ConvolutionalLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				Data& data, Mapped_data& mapped_data);
		//ConvolutionalLayer();

		void setup(Data& data, Mapped_data&, shared_ptr<Layer> prev);

		void forward(bool display);
		void backward();
		void update_weights(Solver& s);

		void test_preparation();

		//weights
		/*
		   const Weights* get_weights() const { return &_weights; }
		   Weights* get_weights() { return &_weights; }
		   */

		void save_state();
		void load_state();

	private:

		ConvolutionParameter _convolution_parameter;

		/* 
		 *	Using DataPackage instead for weights for convolutional layer
		 *	so that it can be used in one skepu call, like the input
		 *  dim0 = Filter width
		 *  dim1 = filter height
		 *  dim2 = filter channels (mainly for colored images.)
		 *  dim3 = filters.
		 */
		DataPackage _weights;
		/*
		 * the change in weights from previous iteration
		 * this is used with when updating the variables with momentum
		 */
		DataPackage _weights_delta;

        //used for storing the temporary sum of weight updates during update phase
        DataPackage _weights_sum;

		skepu2::Vector<double> _bias_weights;
		/*
		 * the change in weights from previous iteration
		 * this is used with when updating the variables with momentum
		 */
		skepu2::Vector<double> _bias_weights_delta; 
		double _bias_values;

		//skepu functionallity 
		SkePU_Convolutional _skepu_f;

		//vars necessary for multiple functions/steps

		skepu2::Vector<size_t> _output_dims;

		//input 
		shared_ptr<DataPackage> _bottom;
		shared_ptr<DataPackage> _delta_top;
		
		//output
		shared_ptr<DataPackage> _top;
		shared_ptr<DataPackage> _delta_bottom;
};
#endif
