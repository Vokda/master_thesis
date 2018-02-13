#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H
#include <skepu2.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <memory>
using namespace std;

#ifndef DEBUG
#define NDEBUG
#endif

//////////////////////////ENUM CLASSES///////////////////////////

enum ActivationType
{
	NO_ACTIVATION, //0 aka identity 
	SIGMOID_ACTIVATION, //1
	RELU_ACTIVATION,  //2
	TANH_ACTIVATION //3
};

//data sets supported by the program
//this will determine template parameter for the layers created
enum class DataSet
{
	UNKNOWN,
	LABEL,
	MNIST,
	CIFAR,
	TEST
	//IMAGE_NET
};

//type of data to be used in the network
enum class DataType
{
	NOT_SUPPORTED,
	MONOCHROME_IMAGE,
	COLOURED_IMAGE,
	DOUBLE,
	INT
};

//weight distributions
enum class WeightDistributions
{
	NOT_SUPPORTED,
	UNIFORM,
	XAVIER
};

/*
//structure used for image data containing coloured data
struct RGB
{
	RGB(double pr=0, double pg=0, double pb=0)
	{
		r=pr;
		g=pg;
		b=pb;
	}
	double r;
	double g;
	double b;
};
*/

//enum for the different types of layers supported
enum LayerType
{
	NOT_A_TYPE,
	INPUT,
	FULLY_CONNECTED,
	EUCLIDIEAN,
	RELU,
	SIGMOID,
	POOL,
	CONVOLUTIONAL,
	SOFT_MAX,
	TANH
};

/////////////////////DATA STRUCTURE////////////////////////////

/*
 * This is the data structure.
 * It contains all the data of a batch as a single skepu vector.
 */
struct DataPackage
{
	//constructors
	DataPackage(skepu2::Vector<size_t>& d, const string& name);

	DataPackage(skepu2::Vector<size_t>&& d, const string& name);

	DataPackage(const string& name);

    DataPackage(const DataPackage& d, const string& copy_name);

    ~DataPackage();
	//functions
	//! get the spatial size of one image (rows * cols)
	size_t get_image_spatial_size() { return _dimensions[0] * _dimensions[1]; }

	//! returns the total size of one image (rows * cols * channels)
	size_t get_image_size() { return get_image_spatial_size() * _dimensions[2]; }

	size_t get_batch_size() { return _dimensions[3]; }

	size_t get_total_size() 
	{
		return accumulate(_dimensions.begin(), _dimensions.end(), 1, std::multiplies<size_t>()); 
	}

	size_t& operator()(size_t index)
	{
		return _dimensions(index);
	}

	size_t& operator[](size_t index)
	{
		return _dimensions[index];
	}

    bool operator==(const DataPackage& other)
    {
        return _data == other._data /*&& _name == other._name*/ && _dimensions == other._dimensions;
    }

	//! not neccessary to use but can make things simpler so that user does not forget to set _data size;
	void set_dimensions(const skepu2::Vector<size_t>& dims, double value=0); 

	//not working properly?
	skepu2::Vector<size_t>& get_dimensions() { return _dimensions; }

	//returns true if the spatial size of the data is square=-shaped
	bool is_square() { return _dimensions[0] == _dimensions[1]; }

	typedef skepu2::Vector<double>::iterator itr;
	//copy from iterator begin to end of another skepu vector
	void copy(itr begin, itr end)
	{
		for(itr i = begin, j = _data.begin(); i != end; ++i, ++j)
		{
			(*j) = (*i);
		}
	}

	//variables

	//! one batch of data
	skepu2::Vector<double> _data;

	string _name;

	void print_indexed_data(skepu2::Vector<double>::iterator begin, skepu2::Vector<double>::iterator end); 
    //should only be used on image data
    void print_image();

    void print_short_info();


	//for printing the datapackage
	friend std::ostream& operator<< (std::ostream& o, DataPackage& d)
	{
		o << "DataPackage " << d._name << endl;
		o << d._name << "._data size: " << d._data.size() << endl;
		size_t dim_size = d.get_dimensions().size();
		size_t total_size = 1;
		cout << "dims size = " << dim_size << endl;
		o << "Dimensions: ";
		for(size_t i = 0; i < dim_size; ++i)
		{
			total_size *= d[i];
			o << d[i];
			if(i < dim_size - 1)
				o << " X ";
		}
		o << endl;
		o << "Total size: " << total_size << endl;
#if DEBUG>1
		//2000 = 80*25 nr of characters that most terminals can display.
		//and to avoid spamming to much data that will not be viewed anyways
		if(total_size > 300) //300 seems more reasonable when printing doubles
		{
			o << "Too much data to display!" << endl;
			return o;
		}
        if(total_size == 1)
        {
            o << "No data to print!" << endl;
            return o;
        }
		auto& dims = d.get_dimensions();
		//for when the vector is 1 dimensionsal (minus the batch dimension)
		if(dims[0] == 1 and dims[1] == 1) 
		{
			o << "one row and column only." << endl;
			for(size_t i = 0; i < dims[3]; ++i) //for each batch
			{
				o << "image " << i << " in batch." << endl;
				for(size_t j = 0; j < dims[2]; ++j) //for each data
				{
					o << d._data[j + (dims[2] * i)];
					o << ' ';
				}
				o << endl;
			}
		}
		else
		{
			o << "size of data  " << d._data.size() << endl;
			for(size_t i = 0; i < dims[3]; ++i) //for each batch
			{
				o << "image " << i << " in batch." << endl;
				size_t img_size = d.get_image_size();

				if(dims[2] == 1) //if there is only one color
				{
					size_t img_size = d.get_image_size();
					for(size_t j = 0; j < img_size; ++j)
					{
						o << ceil(d._data[j + i * img_size]) << ' ';
						if((j+1) % dims[1] == 0)
							o << endl;
					}
					o << endl;
				}
				else // if there are multiple colors
				{
					//for each color
					for(size_t col = 0; col < dims[2]; ++col)
					{
						o << "color " << col << " in image " << i << endl;

						for(size_t rows = 0; rows < dims[1]; ++rows) //rows
						{
							for(size_t cols = 0; cols < dims[0]; ++cols) //cols
							{
								size_t index = cols + 
									rows * dims[1] + 
									col * d.get_image_spatial_size() +
									i * d.get_image_size();  
								//cout << "[index " << index << "] ";
								o << d._data[index] << ' ';
							}
							o << endl;
						}
						o << endl;
					}
					o<< endl;
				}
			}
		}
#endif
		return o;
	}

	private:
	DataPackage() = delete;

	/*
	 * _dimensions[0] = width
	 * _dimensions[1] = height
	 * _dimensions[2] = channels (colours)
	 * _dimensions[3] = batch size (filters if for weights)
	 */
	skepu2::Vector<size_t> _dimensions;
};

///////////TYPE DEFS/////////
typedef skepu2::Matrix<double> Weights;
typedef std::map<string, shared_ptr<DataPackage>> Mapped_data;
typedef std::vector<shared_ptr<DataPackage>> Data;

/*
//typdefs for data types
typedef skepu2::Vector<double> MonochromeImage;
//note: differences between processed images and processed image
typedef skepu2::Vector<double> ProcessedImage;
typedef vector<skepu2::Vector<double>> ProcessedImages;

//typedef skepu2::Matrix<RGB> MonochromeImage;
typedef skepu2::Vector<double> Label;
typedef skepu2::Vector<double> Delta;

 * for simplifying usage of the most common types of batches used
 * Batch_double and Batch_mono are the same type will use different names to different them apart
 * for when easier reading and understanding of the code.
 typedef vector<skepu2::Vector<double>> BatchFloat;
 typedef vector<MonochromeImage> BatchMono;

 template<typename T>
 using Batch = skepu2::Vector<T>;
 */
#endif
