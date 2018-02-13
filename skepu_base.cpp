#include "skepu_base.hpp"
#include <iostream>
#include <cmath>
#include <sstream>
#include <numeric>

using namespace std;

/*
 * global user function defintions
 * TODO see SkePU class declaration comments why
 */

//non-template functions
[[skepu::userfunction]]
double add_f(double a, double b)
{
	return a + b;
}

[[skepu::userfunction]]
double sub_f(double a, double b)
{ 
    return a - b;
}

[[skepu::userfunction]]
double mul_f(double a, double b)
{
	return a * b;
}

[[skepu::instance]]
static auto skepu_sub = skepu2::Map<2>(sub_f);

[[skepu::instance]]
static auto multiplication = skepu2::Map<2>(mul_f);

[[skepu::instance]]
static auto summerize = skepu2::Reduce(add_f);

[[skepu::instance]]
static auto skepu_addition = skepu2::Map<2>(add_f);

[[skepu::userfunction]]
double abs_sum(double a, double b)
{
	return sqrt(pow(a, 2)) + sqrt(pow(b, 2));
}

[[skepu::instance]]
static auto avg_error = skepu2::Reduce(abs_sum);


[[skepu::userfunction]]
double mvp(skepu2::Index1D row, const double* weight, const double* input, const size_t input_size)
{
	double result = 0;
	for (size_t i = 0; i < input_size; ++i)
	{
		result += weight[row.i * input_size + i] * input[i];
	}
	return result;
}

[[skepu::instance]]
static auto mvprod = skepu2::Map<0>(mvp);



/*
[[skepu::userfunction]]
double update_w(skepu2::Index2D out_matrix, double weight, const double* delta, const double* input, const double learning_rate)
{
	return weight + (delta[out_matrix.row] * input[out_matrix.col]  * learning_rate);
}
*/


[[skepu::instance]]
static auto skepu_dot_product = skepu2::MapReduce<2>(mul_f, add_f);

/*
[[skepu::userfunction]]
bool is_negative(double v)
{
	return v < 0;
}

[[skepu::instance]]
static auto negative_element = skepu2::Map<1>(is_negative);
*/

[[skepu::userfunction]]
double sub_scalar(double v, const double s)
{
	return v - s;
}

[[skepu::instance]]
static auto skepu_subtract_scalar = skepu2::Map<1>(sub_scalar);


/***************************
 *	SkePU class definitions
 ****************************/



double SkePU::sum(skepu2::Vector<double>& v)
{
    return summerize(v);
}

void SkePU::addition(skepu2::Vector<double>& a, skepu2::Vector<double>& b, skepu2::Vector<double>& out)
{
	assert(a.size() == b.size());
	skepu_addition(out, a, b);
}

void SkePU::matrix_vector_mul(Weights& w, skepu2::Vector<double>& v, skepu2::Vector<double>& result)
{
	assert(w.total_cols() == v.size());
	assert(result.size() == w.total_rows());
	mvprod(result, w, v, v.size());
}

double SkePU::dot_product(skepu2::Vector<double>& v, skepu2::Vector<double>& w)
{
	assert(v.size() == w.size());
	return skepu_dot_product(v, w);
}

double SkePU::average(skepu2::Vector<double>& v)
{
	return summerize(v) / v.size();
}

double SkePU::average(skepu2::Matrix<double>& v)
{
	return summerize(v) / v.size();
}

/*
vector<double> SkePU::average_error(Batch<Delta>& deltas)
{
	vector<double> image_avg_error(deltas.size());
	for(size_t i = 0; i < deltas.size(); ++i)
	{
		image_avg_error[i] =  avg_error(deltas[i]);
	}
	return image_avg_error;
}
*/

[[skepu::userfunction]]
double max_f(double a, double b)
{
	return max(a, b);
}

[[skepu::instance]]
static auto skepu_find_max = skepu2::Reduce(max_f);

double SkePU::find_max(skepu2::Vector<double>& v)
{
    //if(v.size() <= 10)
    //{
        skepu_find_max.setBackend(skepu2::BackendSpec(skepu2::Backend::typeFromString("cpu")));
    //}
    //else
    //{
        skepu_find_max.setBackend(*_backend_specification);
    //}
    return skepu_find_max(v);
}

/*
bool SkePU::contains_negative_elements(skepu2::Vector<double>& v)
{
	//TODO boolean vectors does not work
	//skepu2::Vector<bool> neg(v.size(), false);
	//negative_element(neg, v);
	for(double e: v)
	{
		if(e < 0)
			return true;
	}
	return false;
}
*/

///SUBTRACTION DEFINITIONS

void subtract_scalar(skepu2::Vector<double>& v, double s)
{
	skepu_subtract_scalar(v, v, s);
}

void SkePU::subtraction(skepu2::Vector<double>& a, skepu2::Vector<double>& b, skepu2::Vector<double>& out)
{
    skepu_sub(out, a, b);
}

///// COPY DATA FROM VECTOR TO VECTOR
[[skepu::userfunction]]
//double cp(skepu2::Index1D index, double a)
double cp(double a)
{
	return a;
};

[[skepu::instance]]
static auto skepu_copy = skepu2::Map<1>(cp);

void SkePU::copy(skepu2::Vector<double>::iterator& from_begin,
		skepu2::Vector<double>& destination)
{
	skepu_copy(destination, from_begin);
}

/////////SUM ABS//////
[[skepu::userfunction]]
double add_abs(double a, double b)
{
	return fabs(a) + fabs(b);
}


[[skepu::instance]]
static auto skepu_sum_abs = skepu2::Reduce(add_abs);

double SkePU::sum_abs(skepu2::Vector<double>& v)
{
	return skepu_sum_abs(v);
}


////////////////////////////////////IM2COL//////////////////////////////////////

/***************************************************************************************
 * restructure data so that it is in proper order for convolution
 * in a matrix matrix style
 * input format: image i's red, green followed by blue color channel. I is followed by the 
 * i+1 image.
 * output format: kernel i's (sub)image (input) red, green followed by blue.
 * kernel i's data is followed by kernel i+n's data for all kernels. This is then followed by 
 * the data for the next image.
 * This is done for the entire batch. 
 */
[[skepu::userfunction]]
double im2col_uf(skepu2::Index1D index, 
		const double* input,
		const size_t* input_dims, 
		const size_t image_spatial_size,
		//const size_t out_image_size,
		const size_t image_size, //size of the input image
		const size_t kernel_size_1d,
		const size_t kernel_size_2d,
		const size_t stride,
		const size_t nr_neurons,
		const size_t nr_neurons_per_row,
		const size_t mul_col_kernel_size_2d,
		const size_t mul_col_nrneurons_k2,
        const size_t image_index) //used to determine which image in batch is being processed
		//const size_t image_index)
{
	//col index (which color channel is being copied//kernel_index / nr_neurons;
	// index.i / size of kernel % color channels
	//size_t image_index = iteration;//index.i / out_image_size;
	size_t width = input_dims[0];
    //size_t t_stride = stride == 1 ? stride : (stride-1)*width;
	size_t color_channels = input_dims[2];
	size_t color_index = (index.i /  kernel_size_2d) % color_channels;

	//which kernel's data is being copied //index.i / kernel_size_2d;
	//(kernel_size_2d * color_channels)
	size_t kernel_index = (index.i / mul_col_kernel_size_2d) ;//% nr_neurons;

	//which row of kernels is being copied
	size_t kernel_index_row = (kernel_index / nr_neurons_per_row) % nr_neurons_per_row;
	
	/* row in kernel
	 * i / kernel width to get row in kernel
	 * % kernel height to not get row outside of kernel.
	 */
	size_t row_kernel = (index.i / kernel_size_1d) % kernel_size_1d;

    kernel_index %= nr_neurons;
	//image index. Which image is being copied
	//(color_channels * nr_neurons * kernel_size_2d)
	//size_t image_index = index.i / mul_col_nrneurons_k2;

	//index to be copied
	size_t copied_index = (image_size * image_index) + //which image
				(image_spatial_size * color_index) + //which color
                (kernel_index_row * (kernel_size_1d - stride))  + //which row of kernels 
                (kernel_index * stride) + //which kernel
				(row_kernel * width) + //which row in a kernel
                (index.i % kernel_size_1d); //which element in the row
#if DEBUG > 1 
	cout << "index " << index.i << endl;
	cout << "image index " << image_index << endl;
	cout << "color index " << color_index << endl;
	cout << "kernel index " << kernel_index << endl;
	cout << "kernel index row " << kernel_index_row << endl;
	cout << "row in kernel " << row_kernel << endl;
	cout << "index calculated " <<  copied_index  << endl;
    cout << "copied " << input[copied_index] << "@" << copied_index << " to output[" << index.i << "]" << endl;
	cout << "-------------------" << endl;

	assert(image_index < input_dims[3]);
	//assert(copied_index <out_image_size);
    /*if(index.i % mul_col_kernel_size_2d == 0)
    {
        cout << "press enter to continue" << endl;
        cin.get();
    }*/
#endif

	//kernel_index superflous? no it is not. 
	return input[copied_index];

	// just the way caffe does it, as some inspiration! ((n * K + k) * H + h) * W + w
}

[[skepu::instance]]
static auto skepu_data_restruct = skepu2::Map<0>(im2col_uf);

void SkePU::im2col(DataPackage& output, 
				DataPackage& input,
				size_t kernel_size,
				size_t kernel_size_2d,
				size_t stride,
				size_t nr_neurons,
                size_t i,
				skepu2::BackendSpec* bs)
{
    if(bs != nullptr)
        skepu_data_restruct.setBackend(*bs);

	/*
	 * if the size is incorrect of the temporary data resize it. Should only happen once at the beginning and
	 * possible at the end of the data if the batch size does not fit properly the data set.
	 */
    size_t nr_colors = input.get_dimensions()[2];
    //size: kernel_size ^ 2, color channels, nr_neurons/nr_kernels, batch size
    size_t precalculated_size =  pow(kernel_size, 2) * nr_colors * nr_neurons;

	if(precalculated_size != output.get_total_size())
	{
		output.set_dimensions(skepu2::Vector<size_t>{kernel_size, kernel_size, nr_colors, nr_neurons});
	}


	//color channels * kernel size
	size_t col_k2 = nr_colors * kernel_size_2d;
	//color channels * kernel_size * number of kernels/neurons
	size_t col_nrn_k2 = col_k2 * nr_neurons;

	//auto& d = conv_in.get_dimensions(); 
	//size_t conv_img_size = std::accumulate(d.begin(), d.end()-1, 1, std::multiplies<size_t>());
	skepu_data_restruct(output._data, 
			input._data, 
			input.get_dimensions(), 
			input.get_image_spatial_size(),
			//conv_img_size,
			input.get_image_size(),
			kernel_size,
			kernel_size_2d,
			stride,
			nr_neurons,
			sqrt(nr_neurons), 
			col_k2,
			col_nrn_k2,
            i
			);

}

/////////////////////////////////// PAD ///////////////////////////////////////////

/* 
 * pads the delta vector
 * @padding_rows_top and @padding_rows_bottom refers to the indecies that are padded with 0.
 */
[[skepu::userfunction]]
double pad(skepu2::Index1D index, 
		const double* delta_top, 
		const size_t input_spatial_size,
		const size_t input_size,
		const size_t padding_width, 
		const size_t padding,:
        const size_t both_side_padding,
		const size_t right_side,
		const size_t start_index,
		const size_t end_index,
		const size_t out_spatial_size,
		const size_t out_size,
		const size_t nr_filters
		)
{
	//index for the current image, which pixel is being processed.
	size_t local_index = index.i % out_spatial_size;

	//simply return 0 on padded rows (top and bottom rows)
	if(local_index < start_index || //first rows
			local_index > end_index || //last rows
	//...also return 0 if index.i is at the columns surrounding the delta values.
			local_index % padding_width < padding || //left columns
			local_index % padding_width >= right_side) //right columns
	{
		return 0;
	}
	else //copy data from input
	{
		//which feature is being copied
		size_t feature_map_index = (index.i / out_spatial_size) % nr_filters;
		//which image is being copied.
		size_t image_index = index.i / out_size;

		//local index for the input data to be copied.
		size_t local_input_index = (index.i - start_index) % input_spatial_size;
        size_t row_index = (index.i / padding_width % input_spatial_size) / 2;

        size_t copy_index = (input_spatial_size * feature_map_index) + 
			(input_size * image_index) + 
            (row_index * both_side_padding) + 
			local_input_index;
#if DEBUG>1
        cout << "index.i " << index.i << endl;
        cout << "start index " << start_index << endl;
        cout << "featuremap index " << feature_map_index << endl;
        cout << "image index " << image_index << endl;
        cout << "row index " << row_index << endl;
        cout << "local input index " << local_input_index << endl;
        cout << "copy index " << copy_index << endl;
        cout << "-------------------" << endl;
#endif

		return delta_top[copy_index];
	}
}
	
[[skepu::instance]]
static auto skepu_pad = skepu2::Map<0>(pad);

void SkePU::pad_for_output(DataPackage& padded_data, 
        DataPackage& input,
        DataPackage& output,
        size_t nr_filters,
        size_t kernel_size,
        size_t stride,
        size_t padding
        )
{
    //aka feature maps
    size_t nr_colors = input.get_dimensions()[2];
    size_t input_width = input.get_dimensions()[0];
    size_t output_width = output.get_dimensions()[0];
	/* 
	 * This is to calculate the size of the padded data to be used in backpropagation.
	 * W = Width * Stride + Filter_size - 2*Padding - Stride
	 * That is the size of one dimension of the data.
	 */
    size_t padded_size =  output_width * stride + kernel_size - 2 * padding - stride;
    size_t total_size = padded_size * padded_size * nr_colors;

    //resize if needed
    if(total_size != input.get_total_size())
    {
        _padded_data.set_dimensions(skepu2::Vector<size_t>{
                padded_size,
                padded_size,
                nr_colors,
                input.get_batch_size()});


        _side_pad = (padded_size - input_width)/2;
        assert(padded_size > input_width); //make sure the padded size is larger than input
        //where to start copying: padding * padded_width + padding. This assumes the data is square shaped
        _index_start = _side_pad * _padded_data.get_dimensions()[0] + _side_pad;

        //where to stop copying
        _index_end = padded_data.get_image_spatial_size() - _index_start;

        assert(_index_end > 0 && _index_end > _index_start);
    }

	skepu_pad(padded_data._data, 
			input._data, 
			input.get_image_spatial_size(),
			input.get_image_size(),
			padded_size,
			_side_pad,
            _side_pad*2,
			padded_size-_side_pad,
			_index_start,
			_index_end,
			_padded_data.get_image_spatial_size(),
			_padded_data.get_image_size(),
			nr_filters);
#ifdef DEBUG
    cout << input << endl;
    cout << padded_data << endl;
    //press_enter_to_continue();
#endif 
}


///constructor

SkePU::SkePU(const shared_ptr<skepu2::BackendSpec> spec):
	_backend_specification(spec), 
    //_conv_out("conv_out"),
    _padded_data("padded data")
    //_conv_w("conv_weights_restructured")
{
	auto& be = *_backend_specification;
	summerize.setBackend(be);
	multiplication.setBackend(be);
	skepu_addition.setBackend(be);
	avg_error.setBackend(be);
	mvprod.setBackend(be);
	//activation.setBackend(be);
	skepu_dot_product.setBackend(be);
	skepu_sum_abs.setBackend(be);
	skepu_data_restruct.setBackend(be);
    skepu_copy.setBackend(be);
    skepu_find_max.setBackend(be);
    skepu_sub.setBackend(be);

    /* gpu works just fine.
     * copy is always set to openmp to avoid copying 
     * large amounts of data to the gpu that will not copied anyway
     */
    //skepu2::BackendSpec openmp(skepu2::Backend::typeFromString("cpu"));
    //skepu_copy.setBackend(openmp);
}
