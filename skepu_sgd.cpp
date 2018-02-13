#include "skepu_sgd.hpp"
#include <algorithm>

///////////////////////////////////////conv/////////////////////////////////////////////////////////

/**
 * For a convolutional layer the neurons shares weights. To update a shared weight all of
 * the updates for a weight are to be summerized, then a weight can be updated with the sum of 
 * all of the updates. In this function a weight is updated with all calculated updates.
 */
[[skepu::userfunction]]
double sum_update_uf(skepu2::Index1D index, 
		const double* delta, 
		const double* restructured_image,
        const size_t image_size,
		const size_t nr_filters,
		const size_t nr_neurons)
{
	//in which kernel is the current weight located.
	size_t kernel_index = index.i % nr_neurons;
	//which filter is being used
	size_t filter_index = (index.i / nr_neurons) % nr_filters;

    size_t i_k = image_size * kernel_index;
    size_t i_f = image_size * filter_index;

	double result = 0;
	for(size_t i = 0; i < image_size; ++i)
	{
		result += restructured_image[i + i_k] * delta[i + i_f];
	}
	return result;
}

[[skepu::instance]]
static auto sum_updates = skepu2::Map<0>(sum_update_uf);


/**
 * For a convolutional layer the neurons shares weights. To update a shared weight all of
 * the updates for a weight are to be summerized, then a weight can be updated with the sum of 
 * all of the updates.
 * this function takes the summerized changes of weights and applies the update.
 */

[[skepu::userfunction]]
double update_w_d(skepu2::Index1D index, 
        double weights_sum,
        double weights,
		double weights_delta,
		const size_t nr_filters,
		const size_t nr_neurons,
		const double learning_rate,
		const double momentum
        )
{
	//in which kernel is the current weight located.
	size_t kernel_index = index.i % nr_neurons;
	//which filter is being used
	size_t filter_index = (index.i / nr_neurons) % nr_filters;

	//the change of weights (delta_w)
    double delta_w = weights_sum * learning_rate + (momentum * weights_delta);
	//double delta_w = update_sum + momentum * weights_delta;

	double new_w = weights - delta_w;

	//save the change in weights
	weights_delta = delta_w;

	return new_w;
}

[[skepu::instance]]
static auto apply_update = skepu2::Map<3>(update_w_d);


/*
 * update the bias weights for conv layer
 */

/*
//bias weight update
[[skepu::userfunction]]
double update_w_conv_bias(skepu2::Index1D index, 
		double weight,
		double weight_delta,
		const double* delta,
		const double learning_rate,
		const size_t nr_neurons,
		const double momentum,
		const double bias)
{
	//sum all delta for bias node
	double result = 0;
	for(size_t i = 0; i < nr_neurons; ++i)
	{
		//nr_neurons * index.i to get the correct delta from the correct feature map
		result += delta[i + (nr_neurons*index.i)] * learning_rate * bias;
	}

	//the change of weights (delta_w)
	double delta_w = learning_rate * result * bias;
        //result + momentum * weight_delta;

	*
	 * new weights = old weights + sum of weight changes (shared weights) +
	 * momentum * previous iterations change in weights
	 
	double new_w = weight - delta_w;

	//save the change in weights
	weight_delta = delta_w;

	//new_w = old_w + sum(update_w)
	return new_w;
}

[[skepu::instance]]
static auto skepu_update_weights_conv_bias = skepu2::Map<2>(update_w_conv_bias);
*/


////////////////CONVOLUTIONAL LAYER WEIGHT UPDATE////////////////////////////
void SkePU_SGD::update_weights_conv(DataPackage& weights,
        DataPackage& delta,
        DataPackage& bottom,
        DataPackage& weights_delta,
        DataPackage& weights_sum,
		double learning_rate,
		double momentum,
        size_t nr_neurons,
        size_t stride,
		bool bias,
        double bias_value,
        skepu2::Vector<double>& bias_weights,
		skepu2::Vector<double>& bias_weights_delta,
        DataPackage* conv_in
        )
{
    size_t batch_size = bottom.get_batch_size();
    size_t kernel_size = weights.get_dimensions()[0]; //kernel size in one dimension(width)
    size_t nr_filters = weights.get_batch_size();
#ifdef DEBUG 
	cout << "weights "  << weights << endl;
	cout << "bottom " << bottom << endl;
	cout << "delta " << delta << endl;
	cout << "lr " << learning_rate << endl;
    learning_rate /= batch_size;
    cout << "lr after scaling with batch_size: " <<  learning_rate << endl;
	cout << "momentum " << momentum << endl;
	cout << "kernel_size " << kernel_size << endl;
	cout << "stride " << stride << endl;
	cout << "nr neurons " << nr_neurons << endl;
    cout << "filters " << nr_filters << endl;
    if(conv_in != nullptr)
		cout << "conv_in " << conv_in << endl;
	auto w_copy(weights._data);
#endif

    /*if(batch_size > 1)
    {
        string error = __FUNCTION__;
        error += " weight update for batch_size > 1 not implemented!";
        throw runtime_error(error);
    }*/


	for(size_t i = 0; i < bottom.get_batch_size(); ++i)
	{
        if(batch_size > 1)
        {
            im2col(*conv_in,
                    bottom,
                    kernel_size,
                    pow(kernel_size,2),
                    stride,
                    nr_neurons,
                    i,
                    nullptr);	
        }
#ifdef DEBUG
        conv_in->print_short_info();
        weights.print_short_info();
        weights_sum.print_short_info();

#endif
		//update weights
        //first summerize weight changes
		sum_updates(weights_sum._data, 
				delta._data,
				conv_in->_data,
                conv_in->get_image_size(),
				weights.get_batch_size(), //this is correct for weights only
				nr_neurons); //nr neurons

		//update the bias weights
		/*if(bias)
		{
			skepu_update_weights_conv_bias(bias_weights, 
					bias_weights,
					bias_weights_delta,
					delta._data,
					learning_rate,
					nr_neurons,
					momentum,
					bias_value);
		}*/
#ifdef DEBUG
        cout << "Completed update for one image." << endl;
#endif
	}
    //apply update
    apply_update(weights._data, 
            weights_sum._data,
            weights._data,
            weights_delta._data,
            nr_filters,
            nr_neurons,
            learning_rate,
            momentum);
#ifdef DEBUG
    cout << "finished updating for all images in batch." << endl;
	if(weights._data == w_copy)
		throw runtime_error("Wegihts have not changed!");

	//make sure no weight is nan
	auto& n_w = weights._data;
	for(size_t i = 0; i < n_w.size(); ++i)
	{
		if(std::isnan(n_w[i]) || n_w[i] != n_w[i])
		{
			stringstream ss;
			ss << "Weight[" << i << "] is NAN: " << n_w[i] << endl;
			throw runtime_error(ss.str());
		}
	}
    press_enter_to_continue();
#endif
}


///////////////////////////////////////FULLY CONNECTED////////////////////////////////////////////////////////

//weight update for fully connected layer.
[[skepu::userfunction]]
double update_w(skepu2::Index2D out_matrix, 
		double weight,
		double weight_delta,
		const double* delta,
		const double* input,
		const double learning_rate,
		const double momentum,
		const size_t image_size,
		const size_t batch_size,
		const size_t delta_size)
{
	//sum all the changes in the batch with the average delta given.
	//double result = 0;
	double delta_w = 0;
	double result = 0;
	for(size_t i = 0; i < batch_size; ++i) //for each image
	{
		double delta_value = delta[out_matrix.row + (i * delta_size)]; //col
		double input_value = input[out_matrix.col + (i * image_size)]; //row
		result += delta_value * input_value; 
/*#ifdef DEBUG
		cout << "delta value " <<  delta_value << endl;
		cout << "input value " << input_value << endl;
		cout << "result " << result << endl;
#endif*/
		//the change of weights (delta_w)
	}

#if DEBUG>2
	cout << "updating weight " << out_matrix.row << " x "  << out_matrix.col << endl;
		cout << "result " << result << endl;
#endif
		delta_w = result * learning_rate + (momentum * weight_delta);
/*#if DEBUG>1
		if (delta_w > 0)
		{
			cerr << "delta_w " << delta_w << endl;
			assert(delta_w <= 0);
		}
#endif*/

	/*
	 * new weights = old weights + sum of weight changes (shared weights) +
	 * momentum * previous iterations change in weights
	 */
	double new_w = weight - delta_w;

	//save the change in weights
	weight_delta = delta_w;

	return new_w;
}

[[skepu::instance]]
static auto skepu_update_weights = skepu2::Map<2>(update_w);

//bias weight update for fully connected layer
[[skepu::userfunction]]
double update_w_bias(skepu2::Index1D index, 
		double weight,
		double weight_delta,
		const double* delta,
		const double bias,
		const double learning_rate,
		const double momentum,
		const size_t batch_size,
		const size_t delta_size)
{
	
	double result = 0;
	//the change of weights (delta_w)
	double delta_w = 0;
	size_t image_index = index.i / delta_size;
	for(size_t i = 0; i < batch_size; ++i) //for each image
	{
		//result = delta[index.i + (i * delta_size)] * bias;
		//delta_w += ((-result) * learning_rate) + (momentum * weight_delta);
		result +=delta[index.i + (image_index * delta_size)];
	}

	delta_w = learning_rate * result * bias ;
    //* (momentum * weight_delta) ;//- (delta * bias * learning_rate);
/*#if DEBUG>1
		if (delta_w > 0)
		{
			cerr << "delta_w bias " << delta_w << endl;
			assert(delta_w <= 0);
		}
#endif*/

	/*
	 * new weights = old weights + sum of weight changes (shared weights) +
	 * momentum * previous iterations change in weights
	 */
	double new_w = weight - delta_w;

	//save the change in weights
	weight_delta = delta_w;

	//new_w = old_w + sum(update_w)
	return new_w;
	//return weight + (delta * bias * learning_rate);
}

[[skepu::instance]]
static auto skepu_update_weights_bias = skepu2::Map<2>(update_w_bias);

/******
 * weight update for fully connected layers with potenial bias nodes
 */
void SkePU_SGD::update_weights(Weights& w, DataPackage& d, Weights& w_delta, 
		DataPackage& input, const double lr, 
		const double m, bool bias, double bias_value, 
		skepu2::Vector<double>& bias_weights, skepu2::Vector<double>& bias_weights_delta)
{
#ifdef DEBUG
	cout << "weight size " << w.total_rows() << 'x' << w.total_cols() << endl;
	if(w.size() < 2000)
		cout << w << endl;
	cout << "delta " << d << endl;
	//cout << "input " << input << endl;
	cout << "copying weights for comparison" << endl;
	Weights w_copy(w);
	cout << "bias to update " << bias << endl;
	cout << "learning_rate " << lr << endl;
	cout << "momentum " << m << endl;
	assert(d.get_batch_size() == input.get_batch_size());
	assert(w.total_rows() == d.get_image_size());
	assert(w.total_cols() == input.get_image_size());
#endif

	//update weights
	skepu_update_weights(w, 
			w, 
			w_delta, 
			d._data, 
			input._data, 
			lr, 
			m, 
			input.get_image_size(), 
			d.get_batch_size(),
			d.get_image_size());

	if(bias)
	{
#ifdef DEBUG
		cout << "updating bias weight" << endl;
#endif
		skepu_update_weights_bias(bias_weights, 
				bias_weights,
				bias_weights_delta, 
				d._data,
				bias_value,
				lr,
				m,
				d.get_batch_size(),
				d.get_image_size());
	}

#ifdef DEBUG
	for(size_t i = 0; i < w.size(); ++i)
	{
		if(std::isnan(w[i]) || w[i] != w[i])
		{
			stringstream ss;
			ss << "Weight[" << i << "] is NaN: " << w[i] << endl;
			throw runtime_error(ss.str());
		}
	}

	if(w == w_copy)
	{
		cout << "delta average: " << average(d._data) << endl;
		cout << "input average: " << average(input._data) << endl;
		throw runtime_error("Weights have not been updated");
	}

	//cout << "average weight before update " << average(w_copy) << endl;
	//cout << "average weight after update " << average(w) << endl;
#endif
}

SkePU_SGD::SkePU_SGD(const shared_ptr<skepu2::BackendSpec>& spec):
	SkePU(spec)
{
	auto& be = *_backend_specification;
	skepu_update_weights.setBackend(be);
	//skepu_update_weights_d.setBackend(be);
    apply_update.setBackend(be);
    sum_updates.setBackend(be);
	//skepu_update_weights_conv_bias.setBackend(be);
	//skepu_update_weights_bias.setBackend(be);
	
}
