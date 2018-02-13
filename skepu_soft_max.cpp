#include "skepu_soft_max.hpp"
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <queue>
#include <milli.hpp>
using namespace std;

///some basic user functions
[[skepu::userfunction]]
double add_uf(double a, double b)
{
	return a + b;
}

[[skepu::userfunction]]
double delta_sub(double label, double net_output)
{
    return net_output-label;
}

[[skepu::instance]]
static auto delta_sub_calc = skepu2::Map<2>(delta_sub);


////////////////////////////////////SOFT MAX/////////////////////////////////////////

////all in one softmax user function

//calculates the actual softmax //OUT OF ORDER
[[skepu::userfunction]]
double softmax_uf(skepu2::Index1D index, double* in)
{
    //find max
    const int classes = 10; //hard coded at the moment since the data sets used only have 10 classes.
	double max_value = 0;
	size_t start_index = index.i * classes;
    size_t end_index = start_index + classes;
	for(size_t i = start_index; i < end_index; ++i)
	{
        max_value = max(in[i], max_value);
	}

    //preprocess
    // e^(y - max(y))
    // and sum
	//size_t max_value_index = index.i / classes;
    double sum = 0;
    for(size_t i = start_index; i < end_index; ++i)
    {
        in[i] = exp(in[i] - max_value);
        sum += in[i];
        
    }

    for(size_t i = start_index; i < end_index; ++i)
    {
        in[i] = in[i] / sum;
    }
    /*
     * having a return statement ruins the memory management later down the line
     * the returned value will overwrite the calculated value
     */
    //return in[index.i]; 
}

[[skepu::instance]]
static auto softmax = skepu2::Map<0>(softmax_uf);


//////soft max as multiple skeletons

//preproc
[[skepu::userfunction]]
double preproc_uf(skepu2::Index1D index, double in, const double max)
{
    return  exp(in - max);
}

[[skepu::instance]]
//static auto preproc = skepu2::Map<0>(preproc_uf);
static auto preproc = skepu2::Map<1>(preproc_uf);
//static auto preproc = skepu2::MapReduce<0>(preproc_uf, add_uf);

//softmax
[[skepu::userfunction]]
double sm_uf(double in, const double sum)
{
    return in / sum;
}

[[skepu::instance]]
static auto sm = skepu2::Map<1>(sm_uf);

//for batch size of 1 only
void SkePU_SoftMax::calculate_softmax(DataPackage& input)
{
    double max, sum_soft_max;
#ifdef DEBUG
	cout << "Calculating softmax (batch size " << input.get_batch_size() << ", 4 skeletons.)" << endl;
	cout << "input " << input << endl;
    assert(input.get_batch_size() == 1);
    
    //multiple skeletons version
    DataPackage test = input;
    //cout << "max" << endl;
    max = find_max(test._data);
    //cout << "preproc" << endl;
    preproc(test._data, test._data, max);
    //cout << "sum" << endl;
    sum_soft_max = sum(test._data);
    //cout << "soft max" << endl;
    sm(test._data, test._data, sum_soft_max);
#endif

    //cout << "max" << endl;
    max = find_max(input._data);
    //cout << "preproc" << endl;
    preproc(input._data, input._data, max);
    //cout << "sum" << endl;
    sum_soft_max = sum(input._data);
    //cout << "soft max" << endl;
    sm(input._data, input._data, sum_soft_max);
    /*
    //sequential version
    input._data.updateHost();
    cout << "seq soft max " << endl;
    //this works assuming the previous layer outputs an image size equal to the number of classes
    size_t classes = input.get_image_size(); 
    input._data.updateHost();
    skepu2::Vector<double>& net_output = input._data;
    //find max
	double max = 0;
	size_t start_index = 0;
    size_t end_index = 0+classes;
	for(size_t i = start_index; i < end_index; ++i)
	{
        //find max value
		if(net_output[i] > max)
			max = net_output[i];
	}

    //preprocess
    // e^(y - max(y))
    // and sum
	//size_t max_value_index = index.i / classes;
    double sum = 0;
    for(size_t i = start_index; i < end_index; ++i)
    {
        net_output[i] = exp(net_output[i] - max);
        sum += net_output[i];
        
    }

	size_t guess = 0;
	double p = 0;
    //actual softmax and correctness
    for(size_t i = start_index; i < end_index; ++i)
    {
        //softmax
        net_output[i] /= sum;
    }
    */

    //map version
    //softmax(input._data, input._data);


#ifdef DEBUG
	cout << "softmaxed (4 skel) " << test << endl;
	cout << "softmaxed (1 skel) " << input << endl;
    //cout << "max " << _max << endl;
    //cout << "sum_sm " << _sum_sm << endl;
    assert(test._data == input._data);

	double s = sum(input._data);
	cout.precision(17);
	cout << "sums of batch: " << fixed << s << endl;
	cout << "sums of batch rounded " << round(s) << endl;
	size_t b =input.get_batch_size();
	size_t ss = round(s);
	assert(ss == b);
	for(auto& d: input._data)
	{
		assert(d >= 0 && d <= 1);
	}
#endif
}

/////////////////////////////////////////////CROSS ENTROPY//////////////////////////////////////
[[skepu::userfunction]]
double cross_entropy_loss_delta_uf(skepu2::Index1D index, double label, double net_output, double* delta)
{
    //cout << "map part" << endl;
    
    delta[index.i] = (net_output-label);
    //cout << "delta: " << delta[index.i] << " = "<< net_output << " - " << label << endl;
    return -label * log(net_output) - (1-label) * log(1-net_output);
}

[[skepu::instance]]
static auto delta_loss_calc = skepu2::MapReduce<2>(cross_entropy_loss_delta_uf, add_uf);

[[skepu::instance]]
static auto delta_loss_calc_map = skepu2::Map<2>(cross_entropy_loss_delta_uf);

[[skepu::userfunction]]
double loss_uf(double label, double net_output)
{
    return -label * log(net_output) - (1-label) * log(1-net_output);
}

[[skepu::instance]]
static auto loss_calc = skepu2::MapReduce<2>(loss_uf, add_uf);
//static auto loss_calc = skepu2::Map<2>(loss_uf);

[[skepu::userfunction]]
double delta_uf(double label, double net_output)
{
    //cout << "delta: " << net_output-label << " = " << net_output << " - " << label << endl;
    return (net_output-label);
}

[[skepu::instance]]
static auto delta_calc = skepu2::Map<2>(delta_uf);


[[skepu::userfunction]]
double sum_per_img_uf(skepu2::Index1D index, double* loss_output)
{
    const int classes = 10;
    double sum = 0;
    for(size_t i = (index.i * classes); i < index.i * classes + classes; ++i)
    {
        sum += loss_output[i];
    }
    return sum;
}

[[skepu::instance]]
static auto sum_per_image = skepu2::Map<0>(sum_per_img_uf);

double SkePU_SoftMax::calculate_cross_entropy(
		DataPackage& target,
		DataPackage& soft_maxed_data,
		DataPackage& delta,
		bool display)
{
    //calculate loss
    //int iat = get_int_activation(at);
#ifdef DEBUG
    cout << "---Calculating cross entropy---" << endl;
    cout << "soft maxed input: " << soft_maxed_data << endl;
    cout << "target " << target << endl;
    bool all_zero = true;
    for(auto& l: soft_maxed_data._data)
    {
        if (l > 0)
            all_zero = false;
    }
    assert(!all_zero);
    DataPackage d2(delta, "delta_copy");
#endif
    double t1, t2;
    /*
    t1 = milli::GetSeconds();
    delta_calc(delta._data, target._data, soft_maxed_data._data);
    t2 = milli::GetSeconds();
    cout << "delta calc" << t2 - t1 << endl;
    t1 = milli::GetSeconds();
    loss_calc(soft_maxed_data._data, target._data, soft_maxed_data._data); //store result in softmax data to save memory
    t2 = milli::GetSeconds();
    cout << "loss calc" << t2 - t1 << endl;*/
    //t1 = milli::GetSeconds();
    //only calculate loss if displayed to save time.
    /*
    if(display)
    {
        //_loss = delta_loss_calc(target._data, soft_maxed_data._data, delta._data);
        t1 = milli::GetSeconds();
        delta_calc(delta._data, target._data, soft_maxed_data._data);
        //store result in softmax data to save memory
        t2 = milli::GetSeconds();
        cout << "delta calc " << t2 - t1 << endl;
        loss_calc(soft_maxed_data._data, target._data, soft_maxed_data._data);
        t1 = milli::GetSeconds();
        cout << "loss calc " << t1 - t2 << endl;
        size_t batch_size = soft_maxed_data.get_batch_size();
        size_t b_start = _counter * batch_size;
        size_t b_end = b_start + (batch_size+1); 
        sum_per_image(_loss.begin() + b_start, _loss.begin() + b_end, soft_maxed_data._data.begin());
        //cout << "loss " << *_loss.begin() << endl;
        //loss = sum(soft_maxed_data._data);
        t2 = milli::GetSeconds();
        //cout << "sum " << t2-t1 << endl;
        cout << "sum per image " << t2 - t1 << endl;
        _counter++;
        //_loss = -1;
    }
    else
    {
        delta_calc(delta._data, target._data, soft_maxed_data._data);
        //_loss = -1; //not displayed or stored
    }*/
    //t2 = milli::GetSeconds();
    //cout << "loss summary" << t2 - t1 << endl;

    t1 = milli::GetSeconds();
    //softmax(soft_maxed_data._data, soft_maxed_data._data);
    _loss = delta_loss_calc(target._data, soft_maxed_data._data, delta._data);
    //subtraction(soft_maxed_data._data, soft_maxed_data._data, delta._data);
    //delta_sub_calc(delta._data, target._data, soft_maxed_data._data);
    t2 = milli::GetSeconds();
    if(display)
        cout << "delta time " << t2 - t1 << endl;

#ifdef DEBUG
    if(!std::isfinite(_loss))
    {
        cout << "soft maxed input " << soft_maxed_data._data << endl;
        cout << "target " << target._data << endl;
        cout << "loss " << _loss << endl;
        throw runtime_error("loss not finite!");
    }

    delta_calc(d2._data, target._data, soft_maxed_data._data);
    double loss_mr = loss_calc(target._data, soft_maxed_data._data);
    cout << delta << endl;
    cout << d2 << endl;
    cout << "loss " << _loss << endl;
    cout << "loss_mr " << loss_mr << endl;
    assert(delta._data == d2._data);
    assert(_loss == loss_mr);
#endif
    return _loss;
}
////////////////////////////////SOFT MAX AND CORRECTNESS///////////////////////////////////////

//calculates the actual softmax
[[skepu::userfunction]]
int sm_cor_uf(skepu2::Index1D index, double* net_output, double* labels)
{
    //unvectorized label 
    int label = 0;
    //find max
    int classes = 10; //data sets used only contain 10 classes
	double max = 0;
	size_t start_index = index.i * classes;
    size_t end_index = start_index + classes;
	for(size_t i = start_index; i < end_index; ++i)
	{
        //find max value
		if(net_output[i] > max)
			max = net_output[i];

        //unvectorize labels in this loop to skip another loop
		if(labels[i] > 0)
            label = i-start_index;
	}

    //preprocess
    // e^(y - max(y))
    // and sum
	//size_t max_value_index = index.i / classes;
    double sum = 0;
    for(size_t i = start_index; i < end_index; ++i)
    {
        net_output[i] = exp(net_output[i] - max);
        sum += net_output[i];
        
    }

	size_t guess = 0;
	double p = 0;
    //actual softmax and correctness
    for(size_t i = start_index; i < end_index; ++i)
    {
        //softmax
        net_output[i] = net_output[i] / sum;

#if DEBUG>1
		cout << "Class " << i-start_index << ": " << net_output[i] << endl;
#endif
        if(net_output[i] > p)
        {
            guess = i;
            p = net_output[i];
        }
    }

#if DEBUG>1
	cout << "Guess: " << guess << endl;
	cout << "Correct answer: " << label << endl;
	cout << "testing answer ";
	if(guess == label)
	{
		cout << "CORRECT" << endl;
	}
	else
	{
		cout << "WRONG" << endl;
	}
	cout << " -------------------------- " << endl;
    cin.get();
#endif
    return guess == label ? 1 : 0;
	//return guess == labels[image_index] ? 1 : 0;
}

[[skepu::instance]]
static auto sm_cor = skepu2::Map<0>(sm_cor_uf);

void SkePU_SoftMax::softmax_correctness(DataPackage& net_output, DataPackage& labels)
{
#ifdef DEBUG
	cout << "Calculating softmax and correctness" << endl;
	cout << "input " << net_output << endl;
	cout << "labels " << labels << endl;
#endif
    //check if size of _correctness batch is correct size
    /*if(_correct.size() != net_output.get_batch_size())
    {
        _correct.resize(net_output.get_batch_size());
    }*/

    double t1, t2;
    cout << "sm_cor skeleton" << endl;
    t1 = milli::GetSeconds();
    //sm_cor(_correct, net_output._data, labels._data);
    t2 = milli::GetSeconds();
    cout << "sm cor time " << t2 - t1 << endl;

    cout << "print data" << endl;
    t1 = milli::GetSeconds();
    /*_correct.updateHost();
	for(size_t i = 0; i < _correct.size(); ++i)
	{
		cout << (_correct(i) == 1 ? "CORRECT" : "WRONG") << endl; //only part needed
	}
    t2 = milli::GetSeconds();
    cout << "print data + update time " << t2 - t1 << endl;
    */
#ifdef DEBUG
	double s = sum(net_output._data);
	cout.precision(17);
	cout << "sums of batch: " << fixed << s << endl;
	cout << "sums of batch rounded " << round(s) << endl;
	size_t b =net_output.get_batch_size();
	size_t ss = round(s);
	assert(ss == b);
	for(auto& d: net_output._data)
	{
		assert(d >= 0 && d <= 1);
	}
#endif
}

////////////////////////////////HELPER FUNCTIONS////////////////////////////////////////


///////// calls every other function needed

double SkePU_SoftMax::calculate_softmax_loss(
		DataPackage& labels, DataPackage& net_output, DataPackage& delta,
		ActivationType at, bool testing, bool display)
{
    double t1,t2;
	double loss = -1;
	if(testing) //if testing only do softmax and correctness calculation
	{
        /*cout << "softmax_correctness gpu" << endl;
        double s = milli::GetSeconds();
        softmax_correctness(net_output, labels);
        double e = milli::GetSeconds();
        cout << "softmax_correctnes time " << e - s << endl;
        return 0;*/

        /*if(net_output.get_batch_size() > 1)
        {
            cout << "softmax corr gpu" << endl;
            softmax_correctness(net_output, labels);
            throw runtime_error("softmax_correctness() not implemented!");
        }
        else*/ 
        {
            t1 = milli::GetSeconds();
            calculate_softmax(net_output);
            t2 = milli::GetSeconds();
            if(display)
                cout << "sm time " << t2-t1 << endl;

            t1 = milli::GetSeconds();
            correctness(labels, net_output, display);
            t2 = milli::GetSeconds();
            if(display)
                cout << "cor time " << t2-t1 << endl;
        }
	}
	else //training
	{
        double t1 = milli::GetSeconds();
		calculate_softmax(net_output);
        double t2 = milli::GetSeconds();
        if(display)
            cout << "sm time " << t2 - t1 << endl;

        t1 = milli::GetSeconds();
		loss = calculate_cross_entropy(labels, net_output, delta, display);
        t2 = milli::GetSeconds();
        if(display)
            cout << "ce time " << t2 - t1 << endl;
        //loss = softmax_crossentropy(labels, net_output, delta);
#ifdef DEBUG
		correctness(labels, net_output, testing);
        cin.get();
#endif
	}
    //throw runtime_error("stop!");
	return loss;
}



////////////////////////// CORRECTNESS //////////////////////////////

//struct for correctness calculations
/*
 * label does not store the actual label. The labels are stored as a one hot vector,
 * in the end all we have to do is see if the label is a one.
 */
struct [[skepu::usertype]] index_et_value
{
    int label; 
    //size_t label_index;
    //for net_output value and index
    double value; 
    //size_t value_index;
    size_t index;
    volatile index_et_value& operator=(const index_et_value& iev) volatile 
    {
        label = iev.label;
        value = iev.value;
        index = iev.index;
        return *this;
    }

    index_et_value& operator=(const index_et_value& iev) 
    {
        label = iev.label;
        value = iev.value;
        index = iev.index;
        return *this;
    }
};

//unvectorize label
[[skepu::userfunction]]
index_et_value correct_2ar_uf(skepu2::Index1D index, double labels, double net_output)
{
    //map index to label
    index_et_value iv;
    iv.label = labels;
    iv.value = net_output;
    iv.index = index.i;
/*#if DEBUG>1
    cout << "val " << iv.value << endl;
    cout << "index " << iv.index << endl;
#endif*/
    return iv;
}

//unvectorize label
[[skepu::userfunction]]
index_et_value correct_uf(skepu2::Index1D index, const double* labels, const double* net_output)
{
    //map index to label
    index_et_value iv;
    iv.label = labels[index.i];
    iv.value = net_output[index.i];
    iv.index = index.i;
/*#if DEBUG>1
    cout << "val " << iv.value << endl;
    cout << "index " << iv.index << endl;
#endif*/
    return iv;
}

[[skepu::userfunction]]
index_et_value corr_red(index_et_value a, index_et_value b)
{
    if(a.value > b.value)
    {
        return a;
    }
    else
    {
        return b;
    }
}

[[skepu::instance]]
static auto correctness_calc = skepu2::MapReduce<2>(correct_2ar_uf, corr_red);

[[skepu::instance]]
static auto map_indexes = skepu2::Map<0>(correct_uf);
[[skepu::instance]]
static auto reduce_indexes = skepu2::Reduce(corr_red);

[[skepu::userfunction]]
int cor_uf(skepu2::Index1D index, double* net_output, double* labels)
{
    //unvectorized label 
    int label = 0;
    //find max
    int classes = 10; //data sets used only contain 10 classes
	size_t start_index = index.i * classes;
    size_t end_index = start_index + classes;
	for(size_t i = start_index; i < end_index; ++i)
	{
        //unvectorize labels in this loop to skip another loop
		if(labels[i] > 0)
            label = i-start_index;
	}

	size_t guess = 0;
	double p = 0;
    //actual softmax and correctness
    for(size_t i = start_index; i < end_index; ++i)
    {
#if DEBUG>1
		cout << "Class " << i-start_index << ": " << net_output[i] << endl;
#endif
        if(net_output[i] > p)
        {
            guess = i;
            p = net_output[i];
        }
    }

#if DEBUG>1
	cout << "Guess: " << guess << endl;
	cout << "Correct answer: " << label << endl;
	cout << "testing answer ";
	if(guess == label)
	{
		cout << "CORRECT" << endl;
	}
	else
	{
		cout << "WRONG" << endl;
	}
	cout << " -------------------------- " << endl;
    cin.get();
#endif
    return guess == label ? 1 : 0;
	//return guess == labels[image_index] ? 1 : 0;
}


[[skepu::instance]]
static auto correctness_skepu = skepu2::Map<0>(cor_uf);

void SkePU_SoftMax::correctness(DataPackage& labels, DataPackage& soft_maxed_data, bool testing)
{
#ifdef DEBUG
	cout << "---Calculating correctness---" << endl;
    assert(soft_maxed_data.get_batch_size() == 1);
#endif


#ifdef DEBUG
	assert(labels.get_image_size() == soft_maxed_data.get_image_size());
	cout << setprecision(1) << labels <<setprecision(10)  << endl;
	cout << soft_maxed_data << endl;
#endif
        index_et_value r = correctness_calc(
                labels._data,
                soft_maxed_data._data);

        if(r.label == 1)
            cout << "CORRECT" << endl;
        else
            cout << "WRONG" << endl;
    /*
    size_t batch_size = labels.get_batch_size();
    size_t b_start = _counter * batch_size; //10 classes
    size_t b_end = b_start + (batch_size); //10 classes
    correctness_skepu(_correct.begin() + b_start, _correct.begin() + b_end,
            soft_maxed_data._data,
            labels._data);
    _counter++;*/
#ifdef DEBUG
    cout << "index " << r.index << endl;
    cout << "probability " << r.value << endl;
    cout << "label " << r.label << endl;
    assert(r.value == soft_maxed_data._data[r.index]);

    //for the MAP + REDUCE varuant
    cout << "Map + Reduce variant" << endl;
    skepu2::Vector<index_et_value> results(labels.get_image_size());
    map_indexes(results, labels._data, soft_maxed_data._data);
    cout << "printing results" << endl;
    /*for(auto& i: results)
    {
        cout << "index " << i.index << endl;
        cout << "probability " << i.value << endl;
        cout << "label " << i.label << endl;
    }*/
	index_et_value r2 = reduce_indexes(results);
    cout << "index " << r2.index << endl;
    cout << "probability " << r2.value << endl;
    cout << "label " << r2.label << endl;
    assert(r.value == r2.value);
#endif
}

void SkePU_SoftMax::print_and_clear_loss()
{
    /*
    for(double l: _loss)
    {
        cout << "loss " << l << endl;
    }
    cout << _loss << endl;
    */
}

void SkePU_SoftMax::print_correctness()
{
    /*
    for(int c: _correct)
    {
        switch(c)
        {
            case 0:
                cout << "WRONG" << endl;
                break;
            case 1:
                cout << "CORRECT" << endl;
                break;
            case -1:
            default:
                cout << "Invalid value: " << c << endl;
                break;
        }
    }
    cout << _correct <<endl;
    */
}

SkePU_SoftMax::SkePU_SoftMax(const shared_ptr<skepu2::BackendSpec>& spec, 
        const SolverParameter& sp):
	SkePU(spec)//, _correct(sp.test_iter(0), -1), _loss(sp.max_iter()/sp.display(), -1)
{
	auto& be = *spec;
    skepu2::BackendSpec cpu(skepu2::Backend::typeFromString("cpu"));
    //softmax
    preproc.setBackend(cpu);
    sm.setBackend(cpu);

    //parallel per image
    softmax.setBackend(be);
    sum_per_image.setBackend(be);
    sm_cor.setBackend(be);

    //entropy
    delta_loss_calc.setBackend(be);
    delta_calc.setBackend(be);
    loss_calc.setBackend(be);
    delta_sub_calc.setBackend(be);

    //helper functions
	correctness_calc.setBackend(be);
    correctness_skepu.setBackend(be);
    //only used in single batch operations at the moment so no parallelism
	//correctness_calc.setBackend(not_gpu);
	//skepu_unvectorize.setBackend(not_gpu);
}
