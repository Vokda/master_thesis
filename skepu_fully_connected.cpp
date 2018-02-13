#include "skepu_fully_connected.hpp"
#include <iomanip>
#include "parser.hpp"

/*
 * Process the entire intput for all neurons
 */
[[skepu::userfunction]]
double forward_neuron(skepu2::Index1D index, const double* input,  
		const double* weights, 
		const size_t img_size,
		const size_t nr_neurons, 
		const double B,
		const int at,
		const double A)
{
	//gives the index of the image.
	size_t image_index = index.i / nr_neurons;

    /*
     * which neurons weights should be accsessed multipled by image size
     * to get correct set of neurons.
     */
	size_t neuron_index = (index.i % nr_neurons) * img_size;

    //image index * image size to obtain corret index in input
    size_t img_index_input = image_index * img_size;

	//For one NEURON calculate the dot product of input and coresponding weight row
	double result = 0;

	for (size_t j = 0; j < img_size; ++j)
	{
		/*
		 * indices:
		 * index.i % nr_neurons == access the correct row of the weight matrix
		 */
		result += weights[neuron_index + j] * input[img_index_input + j];
	}
	
	//apply activation function to result
	switch(at)
	{
		case 1: //logistic
			return 1.0/ (1.0 + exp( ( -B * result ) )); 
		case 2: //relu
			return max(result, 0.0);
		case 3: //tanh
			return tanh(result);
			//A * tanh(B * result);
			 //2.0 / (1.0 + exp(-2.0 * result)) - 1.0;
		default: //identity
			return result;
	}
}

[[skepu::instance]]
static auto forward_batch = skepu2::Map<0>(forward_neuron);

// Forward with bias
[[skepu::userfunction]]
double forward_neuron_bias(skepu2::Index1D index, 
		const double* bias_weight,
		const double* input,  
		const double* weights, 

		const size_t img_size,
		const size_t nr_neurons, 
		const double B, 
		const int at, 
		const double bias_value,
		const double A
		)
{
	//gives the index of the image.
	size_t image_index = index.i / nr_neurons;
	size_t neuron_index = index.i % nr_neurons;
    //image index * image size to obtain corret index in input
    size_t img_index_input = image_index * img_size;
    //multiplied by image size to obtain correct index for weights
    size_t weight_index = neuron_index * img_size;

	//For one NEURON calculate the dot product of input and coresponding weight row
	double result = 0;
	for (size_t j = 0; j < img_size; ++j)
	{
		//result += weights[neuron_index * img_size + j] * input[image_index * img_size + j];
		/*
		 * indices:
		 * index.i % nr_neurons == access the correct row of the weight matrix
		 */
		result += weights[weight_index + j] * input[img_index_input  + j];
	}
	result += bias_weight[neuron_index] * bias_value;

/*#if DEBUG > 1
	cout << "result " << result << endl;
#endif*/
	
	//apply activation function to result
	switch(at)
	{
		case 1: //logistic
			return 1.0/ (1.0 + exp( ( -B * result ) )); 
		case 2: //relu
			return max(result, 0.0);
		case 3: //tanh
			return tanh(result);
			//A * tanh(B * result);
			 //2.0 / (1.0 + exp(-2.0 * result)) - 1.0;
		default: //identity
			return result;
	}
}

[[skepu::instance]]
static auto forward_batch_bias = skepu2::Map<0>(forward_neuron_bias);


/************************************************************************
 * forward defintion
 */
void SkePU_FullyConnected::forward(DataPackage& input, skepu2::Matrix<double>& weights, DataPackage& output,
		const ActivationType& at, bool bias, double bias_value, skepu2::Vector<double>& bias_weights,
		double A, double B)
{
#ifdef DEBUG
	cout << "skepu forward" << endl;
	cout << "weights " << weights.total_rows() << 'x' << weights.total_cols() << endl;
	//if(weights.size() < 2000)
		//cout << weights << endl;
	//cout << "input " << input << endl;
#endif

	size_t img_size = input.get_image_size();
	/*
		skepu2::Index1D index, const double* input, const size_t img_size, 
		const double* weights, const size_t nr_neurons, const double B, const ActivationType& at
		*/
	if(bias)
	{
#ifdef DEBUG
		cout << "fully connected forward with bias" << endl;
		cout << "bias value " << bias_value << endl;
#if DEBUG>1
		//cout << "bias weights " << bias_weights << endl;
#endif
#endif
		forward_batch_bias(output._data, bias_weights, input._data, weights, img_size,
				weights.total_rows(), A, at, bias_value, B);
	}
	else
	{
#ifdef DEBUG
		cout << "fully connected forward" << endl;
#endif
		forward_batch(output._data, input._data, weights, img_size,  weights.total_rows(), A, at, B);
	}
#ifdef DEBUG
	//cout << "output " << output << endl;
#endif
}

//////////////////////////BACKWARD/////////////////////////

/* skepu user function
 *
 *	Each user function will calculate one delta value for one neuron in the previous layer.
 *	
 */
[[skepu::userfunction]]
double error_calc_uf(skepu2::Index1D index, 
		double bottom,
		const double* input_delta,
		const double* weights,
		const size_t delta_in_size, 
		const size_t nr_neurons, 
		const int at,
		const size_t nr_neurons_bottom)
{
	/*
	 * image index calculated by dividing the index of the output by the number of neurons 
	 * in the layer bellow.
	 */
	size_t image_index = index.i / nr_neurons_bottom;


	//compute backprop for one neuron
	double result = 0;
	for (size_t i = 0; i < nr_neurons; ++i)
	{
		result += weights[(i * nr_neurons_bottom) + (index.i % nr_neurons_bottom)] * 
			input_delta[i + (image_index * delta_in_size)]; //does NOT need transpositiong of weight

#ifdef DEBUG
		assert(isfinite(input_delta[i + (image_index * delta_in_size)]));
		assert(isfinite(weights[(i * nr_neurons_bottom) + (index.i % nr_neurons_bottom)]));
#endif
	}

	switch(at)
	{
		case 1: //logistic
			return (bottom * (1.0 - bottom) * result);
		case 2: //relu
			return result <= 0 ? 0 : 1;
		case 3: //tanh
			return (1.0 - pow(tanh(bottom), 2)) * result; 
		default: //identity
			return result;
	}
}

[[skepu::instance]]
static auto error_calc = skepu2::Map<1>(error_calc_uf);



////////////BACKWARD FUNCTION DEFINITION////////
void SkePU_FullyConnected::error_calculation(DataPackage& delta_out, DataPackage& delta_in, 
		skepu2::Matrix<double>& w, ActivationType at, DataPackage& bottom)
{
#ifdef DEBUG
	cout << __FUNCTION__ << endl;
	cout << "input: " << delta_in << endl;
	//cout << "output: " << delta_out << endl;
	cout << "weights " << w.total_rows() << 'x' << w.total_cols() << endl;
	cout << "nr_neurons " << w.total_rows() << endl;
	cout << "activation function to be derived " << Parser::activation_f_to_string(at) << endl;
	//cout << "average in delta " << average(delta_in._data) << endl;
	cout << "bottom " << bottom << endl;

	assert(w.total_rows() == delta_in.get_image_size());
	assert(w.total_cols() == delta_out.get_image_size());
#if DEBUG > 1
	//if(w.size() < 2000)
		//cout << w << endl;
#endif


	assert(bottom.get_image_spatial_size() == delta_out.get_image_spatial_size());
#endif
	error_calc(delta_out._data,
			bottom._data,
			delta_in._data, 
			w,
			delta_in.get_image_size(),
			w.total_rows(),
			at,
			w.total_cols());

}

/*
 *	CONSTRUCTOR
 */

SkePU_FullyConnected::SkePU_FullyConnected(const shared_ptr<skepu2::BackendSpec>& spec):
	SkePU(spec) 
{
	auto& be = * spec;
	error_calc.setBackend(be);
	forward_batch.setBackend(be);
	forward_batch_bias.setBackend(be);
	//map_reduce.setBackend(*_backend_specification);
	//map_activation.setBackend(*_backend_specification);
}
