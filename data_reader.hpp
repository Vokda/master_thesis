#ifndef DATA_READER_H
#define DATA_READER_H
#include <string>
#include "data_structure.hpp"
#include <vector>

using namespace std;
namespace caffe
{
	class TransformationParameter;
}

/*
 * TODO both mnist functions push_back new data instead of preallocating data when size is known
 * resizing vector causes sigsev for some reason. size to large to handle? skepu vector not created properly?
 */

class DataReader
{
	public:
		//data sets supported

		//void read(data_set data, const string& filename, image_data& skepu_data);
		//data info
		//obtains the dimensions from the given file. 
		//vector<unsigned>& get_dimensions() const;

		//mnist data
		/**
		 * reads @elments number of images. If 0 is given all images are read.
		 * Each image is stored in a skepu::Matrix<double>
		 * if the layer above the input layer uses sigmoid activation layer scale data 0 - 1
		 * else scale it -1 - 1
		 */
		void read_mnist_images(DataPackage& imgs, const string& filename, const caffe::TransformationParameter& tp,
				size_t elements = 0) const;
		//void read_mnist_labels(skepu2::Vector<int>& labels, const string& filename) const;
		//void read_mnist_labels(skepu2::Vector<int>& labels, const string& filename, unsigned elements = 0) const;
		/**
		 * labels are stored as vector<skepu::Vector<double>>
		 * label 0 is [1,0,0,0,0,0,0,0,0,0]
		 * label 1 is [0,1,0,0,0,0,0,0,0,0]
		 * etc...
		 */
		void read_mnist_labels(DataPackage& labels, const string& filename, const caffe::TransformationParameter& tp, size_t elements = 0) const;

		/**
		 * read images and labels from the cifar 10 set
		 * elements determines the number of elements to be read. 0 =  all elements.
		 */
		void read_cifar(DataPackage& imgs, DataPackage& labels, bool train, string& data_dir, 
				const caffe::TransformationParameter& tp, size_t elements = 0);


		//will read test data
		void read_test(DataPackage& imgs, DataPackage& labels, bool train, string& data_dir,
				const caffe::TransformationParameter& tp, size_t elements = 0);
	private:
		//This changes endian from high endian to low endian (used by intel processors)
		int reverse_int(int i) const;
		
		/*
		 *	put mnist labels in vector format
		 *	and store the label at index in data
		 */
		void vectorize_label(const int label, int classes, skepu2::Vector<double>& data, const size_t index) const;

		//checks the magic number for the mnist data set
		bool check_magic_nr(const string& filename, int magic_number) const;
};

#endif
