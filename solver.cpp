#include "solver.hpp"
#include <skepu2.hpp>
#include "network.hpp"
#include <milli.hpp>
#include <sstream>

Solver::Solver(SolverParameter sp, const string& backend):
	_solver_parameter(sp), _learning_rate(sp.base_lr())
{
	/*
	 * interpret the flags
	 * If backend is nullptr, that is no argument is given, use best backend possible
	 */
	if(backend.empty()) //no backend arg was given so prototxt backend is selected instead
	{
		const auto& sm = _solver_parameter.solver_mode();
		switch(caffe_to_skepu_backend(sm))
		{
			//TODO enable sequential CPU_SEQ and GPU_CUDA.
			case Solver_mode::GPU_OPENCL:
			case Solver_mode::GPU_CUDA:
				_backend_specification = shared_ptr<skepu2::BackendSpec>(new skepu2::BackendSpec{skepu2::Backend::typeFromString("opencl")});
				cout << "GPU mode selected. (opencl is used atm!)" << endl;
				break;

			case Solver_mode::CPU_OPENMP:
			case Solver_mode::CPU_SEQ:
				_backend_specification = shared_ptr<skepu2::BackendSpec>(new skepu2::BackendSpec{skepu2::Backend::typeFromString("openmp")});
				cout << "CPU mode selected. (openmp is used atm!)" << endl;
				break;
			default:
				cout << "Solver: Invalid solver mode selected." << endl;
				_backend_specification = shared_ptr<skepu2::BackendSpec>(new skepu2::BackendSpec{skepu2::Backend::typeFromString("openmp")});
				break;
		}
	}
	else //a backend argument was given
	{
		cout << "Overriding solver parameter, using program argument instead: " << backend << endl;
		_backend_specification = shared_ptr<skepu2::BackendSpec>(new skepu2::BackendSpec{skepu2::Backend::typeFromString(backend)});
	}

	cout << "Backend selected: "<< _backend_specification->backend() <<  endl;

//#ifdef DEBUG
	stringstream ss;
	ss << _backend_specification->backend();
	string b = ss.str();
	string back_spec(b); //to get back_spec the correct size.
	transform(b.begin(), b.end(), back_spec.begin(), ::tolower);
	if(back_spec != backend)
	{
		cout << "WARNING: Backend set by user: " << backend << ". ";
		cout << "Backend selected by progam: " << back_spec  << endl;
	}
//#endif

	//make sure the necessary parameters are set for the solver
	setup_lr_policy();

	//construct network
	_network = unique_ptr<Network>(new Network(_solver_parameter, Solver::_backend_specification));

	//several checks to make sure there parameters needed exist
	if(!_solver_parameter.has_snapshot_prefix())
		throw runtime_error("snapshot prefix not set!");

	if(_solver_parameter.test_iter_size() < 1)
		throw runtime_error("test iter not set!");

}

Solver::~Solver()
{
}


Solver::Solver_mode Solver::caffe_to_skepu_backend(SolverParameter_SolverMode sp_sm) const
{
	using sm = Solver_mode;
	//TODO should take argv to determine more precise solver_mode
	switch(sp_sm)
	{
		case SolverParameter_SolverMode::SolverParameter_SolverMode_CPU:
			return sm::CPU_OPENMP;
		case SolverParameter_SolverMode::SolverParameter_SolverMode_GPU:
			return sm::GPU_CUDA;
		default:
			cout << "Solver: Invalid solver mode selected." << endl;
			throw runtime_error("Invalid solver mode selected");
	}
}

void Solver::solve()
{
	double time = 0;

	cout << "--- Traning network. ---" << endl;

	//milli::Reset();
    double start = milli::GetSeconds();
	train();
	double end = milli::GetSeconds();
	cout << "Training time " << end - start << endl;

	cout  << "--- Training completed. Test phase beginning. ---" << endl;

	//milli::Reset();
    start =  milli::GetSeconds();
	test();
	end = milli::GetSeconds();
	cout << "Test time " << end - start << endl;

	cout << "--- Test completed. ---" << endl;
}

void Solver::learning_rate_update(size_t itr)
{
	double base_lr = _solver_parameter.base_lr();
#ifdef DEBUG
	cout << "base lr " << base_lr << endl;
	cout << "gamma " << gamma << endl;
	cout << "itr " << itr << endl;
	cout << "step size " << _step_size << endl;
#endif
	switch(_lr_policy)
	{
		case LR::STEP:
			_learning_rate = base_lr * pow(_gamma, floor(itr / _step_size));
			break;
		case LR::EXP:
			_learning_rate = base_lr * pow(_gamma, itr);
			break;
		case LR::INV:
			_learning_rate = base_lr * pow((1 + _gamma * itr), (-_power));
			break;
		default:
			_learning_rate = base_lr;
			break;
	}
	assert(_learning_rate >= 0 && _learning_rate <= 1);
	
}

void Solver::setup_lr_policy()
{
	if(!_solver_parameter.has_lr_policy())
		throw runtime_error("\"lr_policy\" parameter not set!");
	else
	{
		if(!_solver_parameter.has_gamma())
			throw runtime_error("\"gamma\" parameter not set!");
		else
			_gamma = _solver_parameter.gamma();


		if("step" == _solver_parameter.lr_policy())
		{
			_lr_policy = LR::STEP;
			if(!_solver_parameter.has_stepsize())
				throw runtime_error("\"stepsize\" parameter not set!");
			_step_size = _solver_parameter.stepsize();
		}
		else if("exp" == _solver_parameter.lr_policy())
			_lr_policy = LR::EXP;
		else if("inv" == _solver_parameter.lr_policy())
		{
			_lr_policy = LR::INV;
			if(!_solver_parameter.has_power())
				throw runtime_error("\"power\" parameter not set!");
			_power = _solver_parameter.power();
		}
		else
			throw runtime_error("learning rate policy not recognized" + _solver_parameter.lr_policy());
	}

}
