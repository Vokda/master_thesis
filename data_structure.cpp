#include "data_structure.hpp"


DataPackage::DataPackage(skepu2::Vector<size_t>& d, const string& name):
		_data(std::accumulate(d.begin(), d.end(), 1, std::multiplies<size_t>()), 0),  _name(name), _dimensions(d)
	{
		if(_data.size() != get_total_size())
			throw runtime_error("DataPackage: difference in size of _data and _dimensions");
	
	}

DataPackage::DataPackage(skepu2::Vector<size_t>&& d, const string& name):
		_data(std::accumulate(d.begin(), d.end(), 1, std::multiplies<size_t>()), 0),  _name(name), _dimensions(d)
	{
		if(_data.size() != get_total_size())
			throw runtime_error("DataPackage: difference in size of _data and _dimensions");
	}

DataPackage::DataPackage(const string& name):
		_name(name)
	{}

DataPackage::DataPackage(const DataPackage& d, const string& copy_name):
        _name(copy_name),
        _dimensions(d._dimensions),
        _data(d._data)
    {}

DataPackage::~DataPackage()
{
    _data.releaseDeviceAllocations();
    _dimensions.releaseDeviceAllocations();
}

void DataPackage::set_dimensions(const skepu2::Vector<size_t>& dims, double value)
{
#ifdef DEBUG
    cout << "resizing " << _name << endl;
#endif
    _dimensions = dims;
    size_t size = 1;
    for(size_t i = 0; i < _dimensions.size(); ++i)
    {
        size *= _dimensions[i];
    }
    _data.resize(size, value);
}

void DataPackage::print_image()
{
    cout << "Image from datapackage " << _name << endl;
    auto& dims = get_dimensions();
    auto old_precision = cout.precision(0);
    cout << std::fixed;
	if(dims[0] == 1 and dims[1] == 1) 
	{
		cout<< "one row and column only." << endl;
		size_t i;
		for(size_t i = 0; i < dims[3]; ++i) //for each image
		{
			cout<< "image " << i << " in batch." << endl;
			for(size_t j = 0; j < dims[2]; ++j) //for each data
			{
                cout<< std::setprecision(0) << round(_data[j + (dims[2] * i)]);
                cout<< ' ';
			}
			cout<< endl;
		}
	}
    else
    {
        for(size_t i = 0; i < dims[3]; ++i) //for each batch
        {
            cout << "image " << i << " in batch." << endl;
            size_t img_size = get_image_size();

            if(dims[2] == 1) //if there is only one color
            {
                size_t img_size = get_image_size();
                for(size_t j = 0; j < img_size; ++j)
                {
                    cout <<  std::setprecision(0) <<round(_data[j + i * img_size]) << ' ';
                    if((j+1) % dims[1] == 0)
                        cout << endl;
                }
                cout << endl;
            }
            else // if there are multiple colors
            {
                //for each color
                for(size_t col = 0; col < dims[2]; ++col)
                {
                    cout << "color " << col << " in image " << i << endl;

                    for(size_t rows = 0; rows < dims[1]; ++rows) //rows
                    {
                        for(size_t cols = 0; cols < dims[0]; ++cols) //cols
                        {
                            size_t index = cols + 
                                rows * dims[1] + 
                                col * get_image_spatial_size() +
                                i * get_image_size();  
                            //cout << "[index " << index << "] ";
                            cout << std::setprecision(0) <<round(_data[index]) << ' ';
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout<< endl;
            }
        }
    }
    cout.precision(old_precision);
}

void DataPackage::print_indexed_data(skepu2::Vector<double>::iterator begin, skepu2::Vector<double>::iterator end)
{
	cout << "DataPackage " << _name << endl;
	cout << _name << "._data size: " << _data.size() << endl;
	size_t dim_size = get_dimensions().size();
	size_t total_size = 1;
	cout << "dims size = " << dim_size << endl;
	cout << "Dimensions: ";
	for(size_t i = 0; i < dim_size; ++i)
	{
		total_size *= _dimensions[i];
		cout << _dimensions[i];
		if(i < dim_size - 1)
			cout << " X ";
	}
	cout << endl;
	cout << "Total size: " << total_size << endl;
//#if DEBUG>1
	auto& dims = get_dimensions();
	//for when the vector is 1 dimensionsal (minus the batch dimension)
	if(dims[0] == 1 and dims[1] == 1) 
	{
		cout<< "one row and column only." << endl;
		size_t i;
		skepu2::Vector<double>::iterator it = begin;
        for(;it != end; ++it)
        {
            cout<< (*it);
            cout<< ' ';
        }
        cout << endl;
		/*for(i = 0; it != end; ++i, ++it) //for each image
		{
			cout<< "image " << i << " in batch." << endl;
			for(size_t j = 0; j < dims[2]; ++j) //for each data
			{
                cout<< _data[j + (dims[2] * i)];
                cout<< ' ';
			}
			cout<< endl;
		}*/
	}
	else
	{
		cout<< "size of data  " << _data.size() << endl;
		size_t i;
		skepu2::Vector<double>::iterator it = begin;
		for(i = 0; it != end ; ++i, ++it) //for each batch
		{
			cout<< "image " << i << " in batch." << endl;
			size_t img_size = get_image_size();

			if(dims[2] == 1) //if there is only one color
			{
				size_t img_size = get_image_size();
				for(size_t j = 0; j < img_size; ++j)
				{
				cout<< _data[j + i * img_size] << ' ';
					if((j+1) % dims[1] == 0)
					cout<< endl;
				}
			cout<< endl;
			}
			else // if there are multiple colors
			{
				//for each color
				for(size_t col = 0; col < dims[2]; ++col)
				{
				cout<< "color " << col << " in image " << i << endl;

					for(size_t rows = 0; rows < dims[1]; ++rows) //rows
					{
						for(size_t cols = 0; cols < dims[0]; ++cols) //cols
						{
							size_t index = cols + 
								rows * dims[1] + 
								col * get_image_spatial_size() +
								i * get_image_size();  
							//cout << "[index " << index << "] ";
						cout<< _data[index] << ' ';
						}
					cout<< endl;
					}
				cout<< endl;
				}
				cout<< endl;
			}
		}
	}
//#endif
	cout << endl << endl;
};

void DataPackage::print_short_info()
{
	cout << "DataPackage " << _name << endl;
	cout << _name << "._data size: " << _data.size() << endl;
	size_t dim_size = get_dimensions().size();
	size_t total_size = 1;
	cout << "dims size = " << dim_size << endl;
	cout << "Dimensions: ";
	for(size_t i = 0; i < dim_size; ++i)
	{
		total_size *= _dimensions[i];
		cout << _dimensions[i];
		if(i < dim_size - 1)
			cout << " X ";
	}
	cout << endl;
	cout << "Total size: " << total_size << endl;
}
