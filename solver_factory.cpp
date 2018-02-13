#include "solver_factory.hpp"
#include "proto/caffe.pb.h"
#include "proto_io.hpp"
#include <algorithm>
#include <string>
#include <sstream>
//solvers
//#include "solver.h"
#include "sgd_solver.hpp"

using namespace caffe;

SolverFactory::SolverFactory():
	_solver_type(Solvers::NOT_SUPPORTED)
{}
	

Solver* SolverFactory::make_solver(const string& filename, const string& backend)
{
	SolverParameter solver_parameter;
	//read the proto solver file
	_io.read_prototxt(filename, &solver_parameter);

	if(!solver_parameter.has_net())
	{
		throw ::runtime_error("Solver: No network specified.");
	}

	//get the input in some format.
	cout << "Net to be constructed according to " << solver_parameter.net() << endl;

	/* 
	NetParameter net_parameter;
	_io.read_prototxt(solver_parameter.net(), &net_parameter);
	if(!net_parameter.layer(0).has_data_param())
	{
		stringstream ss;
		ss << "The first layer " <<  net_parameter.layer(0).name() << " has no data parameter. First layer is assumed to be the input layer!" << endl;
		throw runtime_error(ss.str());
	}
	string data_dir(net_parameter.layer(0).data_param().source());
	string data_s = data_dir.substr(data_dir.find('/')+1);
	*/

	string solver_type = solver_parameter.type();

	//sets the type of solver according to prototxt
	//should be called BEFORE pick solver so that the factory knows which solver to make.
	set_solver_type(solver_type);

	return pick_solver(solver_parameter, backend);
}

void SolverFactory::set_solver_type(string type)
{
	transform(type.begin(), type.end(), type.begin(), ::tolower);
	if(type == "sgd")
	{
		_solver_type = Solvers::SGD;
		cout << "Stochastic gradient descent solver selected." << endl;
	}
	else
	{
		_solver_type = Solvers::NOT_SUPPORTED;
		cout << "SolverFactory: solver type " << type << " not yet supported." << endl;
		exit(EXIT_FAILURE);
	}
}

Solver* SolverFactory::pick_solver(SolverParameter sp, const string& backend)
{
	//if(data_set == "mnist")
	//{
		switch(_solver_type)
		{
			case Solvers::SGD:
				return new SGDSolver(sp, backend);
			case Solvers::NOT_SUPPORTED:
			default:
				cout << "SolverFactory: Solver type not supported." << endl;
				return nullptr;
				break;
		}
	//}
	//else
	//{
	//	cout << "SolverFactory: vector<DataPackage> type for data set " << data_set << " not supported." << endl;
	//	return nullptr;
	//}
}
