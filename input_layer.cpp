#include "input_layer.hpp"
#include "data_reader.hpp"

InputLayer::InputLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
		vector<shared_ptr<DataPackage>>& data, Mapped_data& mapped_data):
	Layer(ml, spec), 
	_all_training_labels(new DataPackage("full training labels")),
	_all_training_images(new DataPackage("full training images")),
	_all_testing_labels(new DataPackage("full testing labels")),
	_all_testing_images(new DataPackage("full testing images")),
	_batch_size(_layer_parameter.data_param().batch_size()),
	_skepu(spec)
    //_batch_begin(_all_training_images->_data.begin()),
    //_end(_all_training_images->_data.begin()),
    //_batch_end(_all_training_images->_data.end())
{
	string data_name = _layer_parameter.data_param().source();
	cout << "Data source :" << data_name << endl;
	_data_set = _parser.get_data_set(data_name);

	//add directory to data string. Assuming all data is found in the data directory.
	_data_dir = "data/" + data_name + '/';

	cout << "Input layer created." << endl;

	//input layer will only use the tops as it does not have any bottoms;
	vector<string> should_be_empty(Layer::get_bottoms());
	if(!should_be_empty.empty())
	{
		cout << "Warning: Input Layer: This layer contains bottom data when it shouldn't." << endl;
	}

	//read data depending on which data set is selected
	switch(_data_set)
	{
		case(DataSet::MNIST):
			setup_monochrome_data();
			break;
		case(DataSet::CIFAR):
			setup_cifar_data();
			break;
		case(DataSet::TEST):
			setup_test_data();
			break;
		default:
			throw runtime_error("InputLayer: Unknown data set.");
			break;
	}
#if DEBUG>1
	cout << "All data read. Press enter to continue." << endl;
	cin.get();
#endif

	//setup batches
	vector<string> data_names = get_tops(); //TODO assumes specific order of top data


	//allocate top data for images and labels batches
	//images

	//dimensions
	skepu2::Vector<size_t> batch_dimensions = _all_training_images->get_dimensions();
	batch_dimensions[3] = _batch_size;
	
	_batch_images = setup_top(data, mapped_data, data_names[0], batch_dimensions);

	//labels
	//BatchFloat& lbls = get_casted_data<Batch>(_all_labels.get());
	batch_dimensions = _all_training_labels->get_dimensions();
	batch_dimensions[3] = _batch_size;
	//cout << "label size " << batch_dimensions[3] << endl;
	_batch_labels = setup_top(data, mapped_data, data_names[1], batch_dimensions);

/*#ifdef DEBUG
	cout << __FUNCTION__ << endl;
	for(auto& i: mapped_data)
	{
		cout << i.first << endl;
		cout << *i.second << endl;
	}
	for(auto& i: data)
	{
		cout << i << endl;
	}
#endif*/
}


void InputLayer::setup(vector<shared_ptr<DataPackage>>& data, Mapped_data& mapped_data, shared_ptr<Layer> next)
{

	/*
	 * allocate delta vector data.
	 * Input layer will not accutally use it but is for the sake of the top layer to 
	 * be able to allocate the appropriate amount of data
	 */
	//vector& all_imgs = get_casted_data<vector<T>>(_all_images);
	//size_t img_size = all_imgs.front().size();
	//should not be important as the input layer will not used it.
	skepu2::Vector<size_t> dims{1, 1, 1, 1};
	for(string& delta_name: get_deltas())
	{
		setup_delta_receiver(data, mapped_data, delta_name, dims);
	}


    //check to make sure that the batch_size fits with the number of iterations
    _max_batch_counter = _all_training_images->get_batch_size();
    size_t counter = 0;
    while(_batch_size * _max_batch_counter > _all_training_images->get_batch_size())
    {
        _max_batch_counter--;
        counter++;
    }
    cout << "skipping the last " << counter << " images as they do not fit the batch size." << endl;
}

/**************************************************************
 * FORWARD DEFINITION
 */
void InputLayer::forward(bool display)
{
    //shared_ptr<DataPackage> images(_testing ? _all_testing_images : _all_training_images);
    DataPackage& images(!_testing ?  *_all_training_images : *_all_testing_images);
    //DataPackage& images(*_all_testing_images);
    copy_batch(images, *_batch_images);  //copy images

	DataPackage& labels(!_testing ?   *_all_training_labels: *_all_testing_labels);
	//DataPackage& labels(*_all_testing_labels);
    copy_batch(labels, *_batch_labels); //copy labels
    //if(copy_batch(labels, _batch_labels))
    _batch_counter++;
    if(_batch_counter > _max_batch_counter)
    {
        _batch_counter = 0;
        //if(_testing)
        _last_batch = true;
	}
#ifdef DEBUG
	cout << "batch sent forwards " << endl;
	//if(_testing)
	{
		cout << setprecision(0);
		cout << *_batch_images << endl;
		cout << *_batch_labels << endl;
		cout << setprecision(9);
        cout << "Press enter to continue." << endl;
		cin.get();
	}
#endif
}

void InputLayer::copy_batch(DataPackage& data, DataPackage& batch)
{
    skepu2::Vector<double>& data_set = data._data; //this does not look very nice
    size_t img_size = data.get_image_size();

    //from where to start the copying
    size_t offset =  (_batch_counter * _batch_size * img_size);
    auto _batch_begin = data_set[0] + offset;
    //size of data to copy
    size_t copy_size = sizeof(double) * img_size * _batch_size;

    //cout << "pointer stuff" << endl;
    double* dest = &batch._data(0);
    const double* source = &data_set(0) + offset;
#ifdef DEBUG
    //iterators used for debug 
    //runs terribly slow for some reason
    /*
    auto _end = _batch_begin + _batch_size * img_size;
    auto _batch_end = _end > data_set.end() ? data_set.end() : _end; //min(diff, all_imgs.end());

    cout << "batch counter " << _batch_counter << endl;
    cout << "_batch_size " << _batch_size << endl;
    cout << *_all_training_images << endl;
    cout << "Current batch size: " << batch.get_batch_size() << endl;
    cout << "batch begin " << _batch_begin -  data_set.begin() << endl;
    cout << "end " << _end -  data_set.begin() << endl;
    cout << "batch end " << _batch_end - data_set.begin() << endl;
    cout << "data set end " << data_set.end() - data_set.begin() << endl;

    DataPackage cpu(batch.get_dimensions(), "cpu");
    DataPackage skepu(batch.get_dimensions(), "skepu");
    DataPackage memcopy(batch.get_dimensions(), "memcopy");
    DataPackage stdcopy(batch.get_dimensions(), "stdcopy");

    cpu.copy(_batch_begin, _batch_end);
    cpu.print_image();
    _skepu.copy(_batch_begin, skepu._data);
    skepu.print_image();
    memcpy(&memcopy._data[0], &*_batch_begin, sizeof(double) * img_size * _batch_size);
    memcopy.print_image();
    //std::copy(&*_batch_begin, &*_batch_end, &*stdcopy._data.begin());
    //stdcopy.print_image();


    assert(cpu == skepu);
    assert(skepu == memcopy);
    assert(stdcopy == memcopy);

    _skepu.copy(_batch_begin, batch._data);
    batch.print_image();
    assert(cpu == batch);
    cin.get();
    */
#endif
    //batch->copy(_batch_begin, _batch_end);
    //auto& b = *_batch_begin;
    //_skepu.copy(_batch_begin, batch._data);
    //batch->print_image();
    //cin.get();
    //necessary so that SkePU does not deallocate data each iteration
    //memcpy(&*batch._data.begin(), &*_batch_begin, copy_size);
    //cout << "copying data" << endl;
    memcpy(dest, source, copy_size);
    //cout << "finished copying data" << endl;
    batch._data.invalidateDeviceData(); //still dealocates data (OPENCL)
}



void InputLayer::backward()
{
	//nothing to be done here.
}


void InputLayer::setup_monochrome_data()
{
	//this is for testing purposes; 0 =  all data, otherwise read data_size many images/labels
#ifdef DEBUG
	size_t data_size = 2;
#else
    size_t max_i = max(_solver_parameter.max_iter(), _solver_parameter.test_iter(0));
	size_t data_size = 0;
    if(max_i < 60000)
    {
        data_size = max_i;
    }
#endif

	if(data_size > 0 && data_size < _batch_size)
	{
		cout << "data size " << data_size << endl;
		cout << "batch size " << _batch_size << endl;
		throw runtime_error
			("Batch size is larger than data available. Consider a larger data set or smaller batch size");
	}

	/*
	 * These data allocations will contain all of the data set. Batch_sized parts of the data will be
	 * copied to the tops allocated before. The actual copied occurs during the forward function.
	 */
	cout << "Allocating space for all images and labels." << endl;

	/*
						////// read training data //////
	*/

	//read the correct files for either testing or training.
	string images_dir, labels_dir;
	//read training images
	images_dir = _data_dir + "train-images-idx3-ubyte";
	//read training labels 
	labels_dir = _data_dir + "train-labels-idx1-ubyte";


#ifdef DEBUG
	cout << "image dir: " << images_dir  << endl;
	cout << "labels dir: " <<labels_dir << endl;
#endif

	const TransformationParameter& tp = _layer_parameter.transform_param();

	//actually reading the data
	cout << "Reading training images..." << endl;
	_data_reader.read_mnist_images(*_all_training_images, images_dir, tp, data_size);
	cout << "done!" << endl;
	cout << "Number of images in total: " << _all_training_images->get_batch_size() << endl;

	cout<< "Readning training labels..." << endl;
	_data_reader.read_mnist_labels(*_all_training_labels, labels_dir, tp, data_size);
	cout << "done!" << endl;
	cout << "Number of labels in total: " << _all_training_labels->get_batch_size() << endl;


	/*
						///////// read test data /////////
	*/

	//read testing images
	images_dir = _data_dir + "t10k-images-idx3-ubyte";
	//read testing labels 
	labels_dir = _data_dir + "t10k-labels-idx1-ubyte";

#ifdef DEBUG
	cout << "image dir: " << images_dir  << endl;
	cout << "labels dir: " <<labels_dir << endl;
#if DEBUG>1
	cout << *_all_training_images << endl;
	cout << *_all_training_labels << endl;
#endif
#endif

	//actually reading the data
	cout << "Reading testing images..." << endl;
	_data_reader.read_mnist_images(*_all_testing_images, images_dir, tp, data_size);
	cout << "done!" << endl;
	cout << "Number of images in total: " << _all_testing_images->get_batch_size() << endl;

	cout<< "Readning testing labels..." << endl;
	_data_reader.read_mnist_labels(*_all_testing_labels, labels_dir, tp, data_size);
	cout << "done!" << endl;
	cout << "Number of labels in total: " << _all_testing_labels->get_batch_size() << endl;
	//_all_testing_images = _all_training_images;
	//_all_testing_labels=_all_training_labels ;
	if(_all_training_images->get_batch_size() != _all_training_labels->get_batch_size() or
            _all_testing_images->get_batch_size() != _all_testing_labels->get_batch_size())
	{
		throw runtime_error("Dataset contains different number of labels and images!");
	}
#ifdef DEBUG
#if DEBUG>1
	cout << *_all_testing_images << endl;
	cout << *_all_testing_labels << endl;
#endif
	bool positive = false;
	for(auto& i: _all_training_images->_data)
	{
		if(i > 0)
		{
			positive = true;
			break;
		}
	}
	if(!positive)
	{
		string s = __FUNCTION__;
		s+=+ ": MNIST data read is all zeros!";
		cout << _all_training_images->_data << endl;
		throw runtime_error(s);
	}
#endif
}

void InputLayer::setup_cifar_data()
{

	//this is for testing purposes; 0 =  all data, otherwise just read in one batch of the entire data set.
#ifdef DEBUG
	size_t data_size = 2;
	if(_batch_size > data_size)
		data_size = _batch_size;
#else
	const size_t data_size = 0;
#endif

	if(data_size > 0 && data_size < _batch_size)
	{
		cout << "data size " << data_size << endl;
		cout << "batch size " << _batch_size << endl;
		throw runtime_error
			("Batch size is larger than data available. Consider a larger data set or smaller batch size");
	}

	/*
	 * These data allocations will contain all of the data set. Batch_sized parts of the data will be
	 * copied to the tops allocated before. The actual copied occurs during the forward function.
	 */
	cout << "Allocating space for all images and labels." << endl;

	////// read data //////
	const TransformationParameter& tp = _layer_parameter.transform_param();

	//read the correct files for either testing or training.
	// training
	cout << "Reading training data." << endl;
	_data_reader.read_cifar(*_all_training_images, *_all_training_labels, true, _data_dir, tp, data_size);
	// testing
	cout << "Reading test data." << endl;
	_data_reader.read_cifar(*_all_testing_images, *_all_testing_labels, false, _data_dir, tp, data_size);

	cout << "done!" << endl;
	cout << "Number of training images in total: " << _all_training_images->get_batch_size() << endl;
	cout << "Number of training labels in total: " << _all_training_labels->get_batch_size() << endl;
	cout << "Number of testing images in total: " << _all_testing_images->get_batch_size() << endl;
	cout << "Number of testing labels in total: " << _all_testing_labels->get_batch_size() << endl;


	if(_all_training_images->get_batch_size() != _all_training_labels->get_batch_size())
	{
		throw runtime_error("Dataset contains different number of labels and images!");
	}

#ifdef DEBUG
	bool positive = false;
	for(auto& i: _all_training_images->_data)
	{
		if(i > 0)
		{
			positive = true;
			break;
		}
	}
	if(!positive)
	{
		string s {__FUNCTION__};
		s += ": CIFAR data read is all zeros!";
		throw runtime_error(s);
	}
#endif
}

void InputLayer::setup_test_data()
{
#if DEBUG>1
	size_t data_size = 100;
#else
	size_t data_size = -1;
#endif

	if(data_size > 0 && data_size < _batch_size)
	{
		cout << "data size " << data_size << endl;
		cout << "batch size " << _batch_size << endl;
		throw runtime_error
			("Batch size is larger than data available. Consider a larger data set or smaller batch size");
	}

	/*
	 * These data allocations will contain all of the data set. Batch_sized parts of the data will be
	 * copied to the tops allocated before. The actual copied occurs during the forward function.
	 */
	cout << "Allocating space for all images and labels." << endl;

	////// read data //////
	const TransformationParameter& tp = _layer_parameter.transform_param();

	//read the correct files for either testing or training.
	// training
	cout << "Reading training data." << endl;
	_data_reader.read_test(*_all_training_images, *_all_training_labels, true, _data_dir, tp, data_size);
	// testing
	cout << "Reading test data." << endl;
	_data_reader.read_test(*_all_testing_images, *_all_testing_labels, true, _data_dir, tp, data_size);

	cout << "done!" << endl;
	cout << "Number of training images in total: " << _all_training_images->get_batch_size() << endl;
	cout << "Number of training labels in total: " << _all_training_labels->get_batch_size() << endl;
	cout << "Number of testing images in total: " << _all_testing_images->get_batch_size() << endl;
	cout << "Number of testing labels in total: " << _all_testing_labels->get_batch_size() << endl;

	if(_all_training_images->get_batch_size() != _all_training_labels->get_batch_size())
	{
		throw runtime_error("Dataset contains different number of labels and images!");
	}
}


void InputLayer::test_preparation()
{
	_testing = true;
	//reset some values
	_last_batch = false;
	_batch_counter = 0;
	/* not used anymore as all data is read now.
	switch(_data_set)
	{
		case(DataSet::MNIST):
			setup_monochrome_data();
			break;
		case(DataSet::CIFAR):
			setup_cifar_data();
			break;
		default:
			throw runtime_error("InputLayer: Unknown data set.");
			break;
	}
	*/
    //check to make sure that the batch_size fits with the number of iterations
    _max_batch_counter = _all_testing_images->get_batch_size();
    size_t counter = 0;
    while(_batch_size * _max_batch_counter > _all_testing_images->get_batch_size())
    {
        _max_batch_counter--;
        counter++;
    }
    cout << "skipping the last " << counter << " images as they do not fit the batch size." << endl;
}
