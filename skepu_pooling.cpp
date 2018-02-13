#include "skepu_pooling.hpp"
#include "proto/caffe.pb.h"

using namespace caffe;

//skepu definitions

/////////////////////////////////////////// FORWARD ////////////////////////////////////

/*
 * max pools forward 
 * processes entire batch in one go
 */
[[skepu::userfunction]]
double max_pool_uf(skepu2::Index1D index, size_t* indicies, const double* input,
		const size_t* input_dims,
		const size_t image_size,
		const size_t image_spatial_size,

		const size_t kernel_size,
		const size_t kernel_size_2d,
		const size_t stride,

		const size_t nr_neurons,
		const size_t nr_neurons_per_row,


		const size_t mul_col_kernel_size_2d
		)
{
	size_t width = input_dims[0];
	size_t color_channels = input_dims[2];
    //if stride is == 1 width is not necessary. 
    size_t t_stride = stride == 1 ? stride : (stride-1)*width;

	size_t color_index = (index.i /  nr_neurons) % color_channels;

	size_t image_index = color_index / color_channels;

	//which row of kernels is being copied
	// modulus the nr_neurons so that entire batches can be handled at once.
	size_t kernel_index_row = (index.i / nr_neurons_per_row) % nr_neurons_per_row;

    //which kernel is being processed
	size_t kernel_index = (index.i % nr_neurons);

	//index for first element in kernel
	size_t in_index = (image_index * image_size) +  //which image
					(image_spatial_size * color_index) + //which color 
					(kernel_index_row * t_stride) + //which kernel in the row
					(kernel_index * stride);

#if DEBUG>1
	cout << "index " << index.i << endl;
	cout << "image index " << image_index << endl;
	cout << "color index " << color_index << endl;
	cout << "row of kernels " << kernel_index_row << endl;
	cout << "kernel index " << kernel_index << endl;
	cout << "in_index " << in_index << endl;
    //cout << "stride " << stride << endl;
#endif

	//find max value
	double max_val = 0;
    /*
     * set the first index in the kernel to the index saved. 
     * Otherwise 0 is set and may overwrite other index saved.
     */
    indicies[index.i] = in_index; 
	for(size_t i = 0; i < kernel_size; ++i) //row
	{
		for(size_t j = 0; j < kernel_size; ++j) //column
		{
			size_t _i = in_index + (i*width) + j;
#if DEBUG>1
			cout << input[_i] << '@' << _i << ", ";
            assert(_i < image_size);
#endif
			if(input[_i] > max_val)
			{
				max_val = input[_i];
				indicies[index.i] = _i;
			}
		}
	}
#if DEBUG>1
	cout << endl;
	cout << "max value: " << max_val << endl;
	cout << "index saved " << indicies[index.i] << " at " << index.i << endl;
	cout << "---------------------------" << endl;
#endif
	return max_val;
}


[[skepu::instance]]
static auto pooling = skepu2::Map<0>(max_pool_uf);

/////////////////////////////////FORWARD//////////////////////////////////

void SkePU_Pooling::forward(DataPackage& in, DataPackage& out, const caffe::PoolingParameter& pp,
		size_t nr_neurons)
{
#ifdef DEBUG
	cout << "pool forward" << endl;
	cout << "input: " << in << endl;
	cout << "number of indicies " << _indicies.size() << endl;
	cout << "nr of neurons " << nr_neurons << endl;
	cout << "kernel size " << pp.kernel_size() << 'x' << pp.kernel_size() << endl;
	cout << "stride " << pp.stride() << endl;
#endif

	//used for calculating the correct image index when processing 
	size_t col_k2 = in.get_dimensions()[2] * pow(pp.kernel_size(), 2);

	pooling(out._data, 
			_indicies, in._data, in.get_dimensions(),
			in.get_image_size(),
			in.get_image_spatial_size(),

			pp.kernel_size(),
			pp.kernel_size() * pp.kernel_size(),
			pp.stride(), 
			nr_neurons,
			sqrt(nr_neurons),
			col_k2
			);
#ifdef DEBUG
	//cout << "output: " << out << endl;
	cout << "indicies saved: " << _indicies << endl;
#endif
}


/////////////////////////////BACKWARD/////////////////////////

[[skepu::userfunction]]
double unpool(skepu2::Index1D index, 
		const double* delta_top,
		const size_t* indicies,
		const size_t indicies_size,
		const size_t kernel_size)
{
	double delta = 0;
	for(size_t i = 0; i < indicies_size; ++i)
	{
		if(indicies[i] == index.i)
		{
			size_t top_index = index.i/kernel_size;
			delta = delta_top[top_index];
			break;
		}
	}
	return delta;
}

[[skepu::instance]]
static auto up_sampling = skepu2::Map<0>(unpool);


/////backward function
void SkePU_Pooling::backward(DataPackage& delta_bottom, DataPackage& delta_top,
		const caffe::PoolingParameter& pp)
{
#ifdef DEBUG
	cout << delta_top << endl;
	cout << delta_bottom << endl;
	cout << "indecies saved: " <<  _indicies << endl;
#endif
	up_sampling(delta_bottom._data, 
			delta_top._data,
			_indicies,
			_indicies.size(),
			pow(pp.kernel_size(), 2));
#ifdef DEBUG
	cout << delta_bottom << endl;
	press_enter_to_continue();
#endif
}

///////////////////////////// CONSTRUCTOR /////////////////////////////

SkePU_Pooling::SkePU_Pooling(const shared_ptr<skepu2::BackendSpec>& spec):
	SkePU(spec)
{
	auto& be = * spec;
	pooling.setBackend(be);
	up_sampling.setBackend(be);

}
