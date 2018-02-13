#ifndef LAYER_INPUT_H
#define LAYER_INPUT_H
#include "layer.hpp"
#include "data_reader.hpp"
#include "parser.hpp"
#include "skepu_base.hpp"
using namespace std;

class InputLayer: public Layer
{
	public:
		InputLayer(const MetaLayer& ml, const shared_ptr<skepu2::BackendSpec> spec,
				vector<shared_ptr<DataPackage>>& data, Mapped_data&);

		void setup(Data& data, Mapped_data&, shared_ptr<Layer> prev);
		void forward(bool display);
		void backward();
		void update_weights(Solver&){};
		void test_preparation();
	private:

		void set_meta_data(string& data);

		//setup function for specific data set types
		//void setup_monochrome_data(vector<DataPackage>& data, map<string, DataPackage*>& mapped_data);
		void setup_monochrome_data();
		void setup_cifar_data();
		void setup_test_data();

		DataSet _data_set;
		string _data_dir;

		DataReader _data_reader;

		//pointers to data and labels
		shared_ptr<DataPackage> _batch_images;
		shared_ptr<DataPackage> _batch_labels;

		shared_ptr<DataPackage> _all_training_labels;
		shared_ptr<DataPackage> _all_training_images;

		shared_ptr<DataPackage> _all_testing_labels;
		shared_ptr<DataPackage> _all_testing_images;

		size_t _batch_size;
		//counts which batch is being copied 
		size_t _batch_counter{0}; 
        //max number of iterations over the data set provided.
        size_t _max_batch_counter = 0;

        //skepu2::Vector<double>::iterator _batch_begin;
        //skepu2::Vector<double>::iterator _end; //temp end necessary for calculations
        //skepu2::Vector<double>::iterator _batch_end;

        void copy_batch(DataPackage& data, DataPackage& batch);
        //void copy_labels();

		//if true read images and lables for testing rather than training
		//bool _testing;
		Parser _parser;

		SkePU _skepu;
};
#endif
