#include "data_reader.hpp"
#include "data_structure.hpp"
#include <iostream>
#include <bitset>
#include <vector>
#include <fstream>
#include <algorithm>
#include "proto/caffe.pb.h"

using namespace caffe;
using namespace std;
//using namespace skepu2;

/*void vector<DataPackage>_reader::read(data_set ds, const string& filename,  image_data& skepu_data)
{
	switch(ds)
	{
		case MNIST:
			read_mnist_images(img, filename);
			break;
		case CIFAR:
		case IMAGE_NET:
		default:
			cout << "No data set recognized." << endl;
			break;
	}
}
*/

int DataReader::reverse_int(int i) const
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int) c1 << 24 ) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void DataReader::read_mnist_images(DataPackage& images, const string& filename, const TransformationParameter& tp,
		size_t elements) const
{
	ifstream file(filename, ios::binary);
	if(file.is_open())
	{
		//imgs.resize(nr_imgs, vector<double>(data_img));
		int magic_number;
		size_t nr_images;
		size_t rows;
		size_t cols;
		
		// all the data is in high endian, which is not what this excess system is using
		// all data needs to be reversed. 

		//first the mnist data contains a 32-bit int (MSB)
		file.read((char*) &magic_number, sizeof(int));
		magic_number = reverse_int(magic_number);
		if(!check_magic_nr(filename, magic_number))
			throw runtime_error("Wrong magic number read from file!");
		//secondly the mnist data contains a 32-bit int that describes the number of images
		file.read((char*) &nr_images, sizeof(int));
		nr_images = reverse_int(nr_images);
		//thridly rows
		file.read((char*) &rows, sizeof(int));
		rows = reverse_int(rows);
		//_dimensions[0] = rows;
		//4th cols
		file.read((char*) &cols, sizeof(int));
		cols = reverse_int(cols);
		size_t max_images = elements == 0 ? nr_images : elements;

		//set dimensions
		images.set_dimensions(skepu2::Vector<size_t>{cols, rows, 1, max_images});
#ifdef DEBUG
		cout << "read magic number " << magic_number << endl;
		cout << "images  " << nr_images << endl;
		cout << "read rows "  << rows << endl;
		cout << "read cols " << cols << endl;
		cout << "data set dimensions" << endl;
		size_t tot = 0;
		for(size_t d: images.get_dimensions())
		{
			cout << d << endl;
			tot += d;
		}
		if(tot > images._data.size())
		{
			cout << "images datapackage not large enough for data set!" << endl;
			cout << "dataset size = " << tot << endl;
			//cout << images << endl;
		}
		//cout << images<< endl;
#endif


		//the rest of the data is just images in row-major order
		//size_t nr_images_to_read = max_images * rows*cols;
		size_t image_size = rows * cols;
		for(size_t i = 0; i < max_images; ++i)
		{
			//for each row
			for(size_t j = 0; j < rows; ++j)
			{
				//for each column
				for(size_t k = 0; k < cols; ++k)
				{
					//calculate index
					size_t index = (image_size * i) + (cols*j) + k;
					//get data point
					unsigned char temp;
					file.read((char*)&temp, sizeof(temp));
					//subtract mean value first (like caffe does) before scaling the data value
					if(tp.mean_value_size() > 0)
					{
						//(1/256) to scale the 0-255 ubyte to 0-1 double
						images._data[index] = ((double)temp - tp.mean_value(0)) * tp.scale();
					}
					else
					{
						//(1/256) to scale the 0-255 ubyte to 0-1 double
						images._data[index] = (double)temp * tp.scale();
					}

					//if(at == ActivationType::TANH_ACTIVATION);
					//scale to -1 to 1 
					//temp_image[index] = ((double)temp) / 127.5 - 1.0;
				}
			}

		}
	}
	else 
	{
		string e = "Data reader: File " + filename + " not found.";
		throw runtime_error(e);
	}
}
		
void DataReader::read_mnist_labels(DataPackage& labels, const string& filename, const TransformationParameter& tp,
		size_t elements) const
{
	const int nr_classes = 10;
	ifstream file(filename, ios::binary);
	if(file.is_open())
	{
		//imgs.resize(nr_imgs, vector<double>(data_img));
		int magic_number;
		size_t nr_labels;
		
		// all the data is in high endian, which is not what this excess system is using
		// all data needs to be reversed. 

		//first the mnist data contains a 32-bit int (MSB)
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		if(!check_magic_nr(filename, magic_number))
			throw runtime_error("Wrong magic number read from file!");
		//secondly the mnist data contains a 32-bit int that describes the number of labels
		file.read((char*) &nr_labels, sizeof(int));
		nr_labels = reverse_int(nr_labels);
		//labels.resize(nr_labels);

		size_t max_labels = elements == 0 ? nr_labels : elements;
		/*
		 * if there are mean value to subtract this means the data range from -1 to 1
		 * instead of 0 to 1 so labels should be -1 to 1 as well. 
		 * NOTE: this is only true for this thesis, this is a quick fix
		 */
		labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, max_labels}, 0);
		/*if(tp.mean_value_size()>0) 
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, max_labels}, -1);
		else
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, max_labels});*/


		//for each label
		for(size_t i = 0; i < max_labels; ++i)
		{
			unsigned char temp;
			file.read((char*)&temp, sizeof(temp));
#ifdef DEBUG
			cout << "label " << int(temp) << endl;
			assert(int(temp) <= 9 && int(temp) >= 0);
#endif
			vectorize_label(temp, nr_classes, labels._data, i);
			//labels._data[i*10] = *vectorize_label((int)temp);
		}
	}
	else 
	{
		string e = "vector<DataPackage> reader: File " + filename + " not found.";
		throw runtime_error(e);
	}
}

/*
skepu2::Vector<double>* DataReader::vectorize_label(const int& label) const
{
	skepu2::Vector<double>* vector_label = new skepu2::Vector<double>(10, 0);
	vector_label->at(label) = 1;
	return vector_label;
}
*/

void DataReader::vectorize_label(const int label, int classes, skepu2::Vector<double>& data, const size_t index) const
{
	data[index * classes + label] = 1;
#ifdef DEBUG
	//unvectorize label to control
	
	int unveclabel = -1;
	for(size_t i = 0; i < classes; ++i)
	{
		if(data[classes * index + i] > 0)
			unveclabel = i;
	}
	//cout << endl;
	if(label != unveclabel)
	{
        for(size_t i = 0; i < classes; ++i)
        {
            cout << data[classes * index + i] << ' ';
        }
        cout << "unvectorized label " << unveclabel << endl;
        cout << "expected label " <<  label << endl;
		throw runtime_error("incorrect label!");
	}
#endif
}

void DataReader::read_cifar(DataPackage& imgs, DataPackage& labels, bool train, string& data_dir,
		const TransformationParameter& tp, size_t elements)
{
	size_t data_size;
	const size_t samples_per_batch = 10000;
	const size_t nr_classes = 10;
	const size_t image_total_size = 32 * 32 * 3;
	if(train)
	{
		vector<string> file_names{"data_batch_2.bin", "data_batch_4.bin," "data_batch_1.bin,"
			"data_batch_3.bin", "data_batch_5.bin"}; 

		data_size = elements == 0 ? 50000 : elements;

		//set sizes
		imgs.set_dimensions(skepu2::Vector<size_t>{32, 32, 3, data_size});
		//labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, data_size});

		if(tp.mean_value_size() > 0)
		{
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, data_size}, -1);
		}
		else
		{
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, data_size});
		}
		

		/*
		 * cifar format:
		 * label (1) image (32 * 32 * 3);
		 */

		/*
		 * index for data in datapackages, global for the different batches read.
		 * i.e.: index = 12000 is the 2000th data sample in the second batch but the 
		 * 12000th data sample in the data package
		 */
		size_t index = 0;
		//for each batch (10000 images and labels in each batch)
		for(size_t i = 0; i < file_names.size(); ++i)
		{

			ifstream file(data_dir + file_names[i], ios::binary);

			if(file.is_open())
			{
				//for each image/label
				for(size_t j = 0; j < samples_per_batch; ++j)
				{
					if(index == data_size) //break if we have read all the data we want to read.
						break;
					//read label
					unsigned char label;
					file.read((char*) &label, sizeof(label));
					vectorize_label(label, nr_classes, labels._data, index);

					//read image
					//for each color
					for(size_t channel = 0; channel < 3; ++channel)
					{
						//for each row
						for(size_t r = 0; r < 32; ++r)
						{
							//for each column
							for(size_t c = 0; c < 32; ++c)
							{
								unsigned char temp = 0;
								file.read((char*) &temp, sizeof(temp));
								//(1/256) to scale the 0-255 ubyte to 0-1 double
								size_t img_index = (index * image_total_size) + (32*r) + c;

								//subtract mean value before scaling data value
								if(tp.mean_value_size() > 0)
								{
									//(1/256) to scale the 0-255 ubyte to 0-1 double
									imgs._data[img_index] = ((double)temp - tp.mean_value(0)) * tp.scale();
								}
								else
								{
									//(1/256) to scale the 0-255 ubyte to 0-1 double
									imgs._data[img_index] = (double)temp * tp.scale();
								}
								//imgs._data[img_index] = (double)temp * (1.0/256.0);
							}

						}
					}

					index++;
				}
			}
			else
			{
				cout << "File " << file_names[i] << " not found!" << endl;
			}

			file.close();
		}
	}
	else //testing
	{
		string file_name = "test_batch.bin";
		data_size = elements == 0 ? 10000 : elements;
		ifstream file(data_dir + file_name, ios::binary);

		//set sizes
		imgs.set_dimensions(skepu2::Vector<size_t>{32, 32, 3, data_size});

		if(tp.mean_value_size() > 0)
		{
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, data_size}, -1);
		}
		else
		{
			labels.set_dimensions(skepu2::Vector<size_t>{1, 1, 10, data_size});
		}


		if(file.is_open())
		{
			//for each image/label
			for(size_t j = 0; j < samples_per_batch; ++j)
			{
				if(j == data_size) //break if we have read all the data we want to read.
					break;
				//read label
				unsigned char label;
				file.read((char*) &label, sizeof(label));
				vectorize_label(label, nr_classes, labels._data, j);

				//read image
				//for each color
				for(size_t channel = 0; channel < 3; ++channel)
				{
					//for each row
					for(size_t r = 0; r < 32; ++r)
					{
						//for each column
						for(size_t c = 0; c < 32; ++c)
						{
							unsigned char temp = 0;
							file.read((char*) &temp, sizeof(temp));
							//calculate index
							size_t img_index = (j * image_total_size) + (32*r) + c;

							//subtract mean value before scaling data value
							if(tp.mean_value_size() > 0)
							{
								//(1/256) to scale the 0-255 ubyte to 0-1 double
								imgs._data[img_index] = ((double)temp - tp.mean_value(0)) * tp.scale();
							}
							else
							{
								//(1/256) to scale the 0-255 ubyte to 0-1 double
								imgs._data[img_index] = (double)temp * tp.scale();
							}
							//imgs._data[img_index] = (double)temp * (1.0/256.0);
						}

					}
				}
			}
		}
		else
		{
			cout << "File " << file_name << " not found!" << endl;
		}

		file.close();
	}
}

void DataReader::read_test(DataPackage& imgs, DataPackage& labels, bool train, string& data_dir,
		const TransformationParameter& tp, size_t elements)
{
	size_t data_set_size = -1; //if data set does not have a size errors will occur later down the line
	size_t nr_classes = -1;
	//size_t image_total_size = 32 * 32 * 3;
	if(train)
	{
		string file_name = "train.txt";

		ifstream file(data_dir + file_name, ios::binary);

		if(file.is_open())
		{
			string line;
			size_t index = 0;

			//read image size
			getline(file, line);
			istringstream iss(line);
			size_t w, h, c; //width, height, colors
			iss >> w >> h >> c >>  data_set_size >> nr_classes;
	
			if(elements < data_set_size && elements > 0)
				data_set_size = elements;
			cout << "width " << w << endl;
			cout << "height " << h << endl;
			cout << "colors " << c << endl;
			cout << "data set size " << data_set_size << endl;
			cout << "number of classes " << nr_classes << endl;

			//set images size
			if(tp.mean_value_size() > 0)
				imgs.set_dimensions(skepu2::Vector<size_t>{w, h, c, data_set_size}, -1);
			else
				imgs.set_dimensions(skepu2::Vector<size_t>{w, h, c, data_set_size});

			//set labels size
			/*if(tp.mean_value_size() > 0)
				labels.set_dimensions(skepu2::Vector<size_t>{1,1,nr_classes, data_set_size}, -1);
			else
				labels.set_dimensions(skepu2::Vector<size_t>{1,1,nr_classes, data_set_size});*/
			labels.set_dimensions(skepu2::Vector<size_t>{1,1,nr_classes, data_set_size});


			size_t img_count = 0;
			while (getline(file, line) && img_count++ < elements)
			{
				istringstream iss(line);
				//read image
				//cout << "image " << endl;
				for(size_t i = 0; i < imgs.get_image_size(); ++i)
				{
					double a;
					iss >> a;
					if(tp.mean_value_size() > 0 )//and a == 0)
						//a =  -1; //a - tp.mean_value(0) * tp.scale();
						a = (a -tp.mean_value(0)) * tp.scale();

					imgs._data[i + (imgs.get_image_size() * index)] = a; 
					//cout << imgs._data[i + (imgs.get_image_size() * index)];
				}
				//cout << endl;
				//read label
				double l;
				iss >> l;
				vectorize_label(l, nr_classes, labels._data, index);
				//cout << "labels" << endl;
				//cout << labels._data << endl;

				index++;
			}
		}
		else
		{
			cout << "File " << file_name << " not found!" << endl;
		}

		//cout << imgs << endl;
		file.close();
		//cin.get();
	}
	else //testing
	{
		throw runtime_error("Debug data set is only for training!");
	}
}

bool DataReader::check_magic_nr(const string& filename, int magic_number) const
{
	if(string::npos != filename.find("train-labels") or 
			string::npos != filename.find("t10k-labels"))
	{
		if(magic_number == 2049)
			return true;
		else
		{
			cout << "file name: " << filename << endl;
			cout << "magic number read: " << magic_number << endl;
			cout << "magic number expected: 2049" << endl;
			return false;
		}
	}
	else if(string::npos != filename.find("train-images") or
			string::npos != filename.find("t10k-images"))
	{
		if(magic_number == 2051)
			return true;
		else
		{
			cout << "file name: " << filename << endl;
			cout << "magic number read: " << magic_number << endl;
			cout << "magic number expected: 2051" << endl;
			return false;
		}
	}
	cout << "file name: " << filename << endl;
	cout << "file name not recognized!" << endl;
	return false;
}
