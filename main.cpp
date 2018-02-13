#include <iostream>
#include "solver.hpp"
#include "solver_factory.hpp"
#include <exception>
#include <milli.hpp>
#include <memory>

#ifdef SKEPU_CUDA
#include <cuda_profiler_api.h>
#endif

#define SKEPU_ENABLE_EXCEPTION
#ifdef DEBUG
#ifndef SKEPU_DEBUG
assert(false);
#endif
#endif
//#define REMOVE_OUTPUT
using namespace std;

int main(int argc, char** argv)
{
	//solver will solve the net
	unique_ptr<Solver> solver;
	//will choose and make an appropriate solver depending on input
	SolverFactory solver_factory;
    double start, end;
	//read input.
	if(argc >= 2)
	{
		try
		{
			milli::Reset();
			string backend;
			if(argc == 3) //if a backend is given
			{
				cout << "Backend set to " << argv[2] << endl;
				backend = argv[2];
			}
			//solver.reset(solver_factory.make_solver(argv[1], backend));
            solver = unique_ptr<Solver>(solver_factory.make_solver(argv[1], backend));

			double constr_time = milli::GetSeconds();

			//set file name to be used by data collector. solverparam not set until solver is set.
			//cout << "Network construction time: " << constr_time << endl;
		}
		catch(const runtime_error& e)
		{
			cout << "NETWORK CONSTRUCTION ERROR: " << e.what() << endl;
			return 1;
		}
	}
	else
	{
		cout << "Wrong number of arguments: " << argc << " expected at least 2." << endl;
		cout << "How to use:" << endl;
		cout << "Input a solver.prototxt file." << endl;
		cout << "The file should contain options about the running about solver of the network." << endl;
		cout << "The file should also contain the relative path to a *.prototxt, which will ";
		cout << "describe the structure of the neural network to be used." << endl;
		cout << "Example: ./skepu_ann something_solver.prototxt" << endl;
		cout << "It is also possible to give an argument for which backend SkePU should use" << endl;
		return 1;
	}

	
	try
	{
		//train and test network and start timer; happens in network
        start = milli::GetSeconds();
#ifdef SKEPU_CUDA
        //cudaProfilerStart();
#endif
		solver->solve(); //solving network
#ifdef SKEPU_CUDA
        //cudaProfilerStop();
#endif
		end = milli::GetSeconds();
        cout << "Network Solving done. Time for solution: " << end  - start << endl;
	}
	catch(const exception& e)
	{
		cout << "NETWORK CALCULATION ERROR: " << e.what() << endl;
		cout << "Occured during" << (solver->training() ? " training " : " testing ") << "iteration " << solver->iteration() << endl;
        cout << "Release resources...";
        solver.reset(nullptr);
        cout << "Completed!" << endl;
		return 1;
	}
    cout << "Release resources..."<<endl;
    solver.reset(nullptr);
    cout << "Completed!" << endl;
    cout << "Good bye!" << endl;
	return 0;
};
