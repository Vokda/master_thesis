#ifndef SOLVER_FACTORY_H
#define SOLVER_FACTORY_H
#include "data_structure.hpp"
#include "proto_io.hpp"
#include <memory>

namespace skepu2
{
	class BackendSpec;
}
class Solver;
using namespace std;

class SolverFactory
{
	public:
		SolverFactory();
		
		/**
		 * Will return an appropriate solver based on input
		 */
		Solver* make_solver(const string& filename, const string& backend);
	private:

		/**
		 * Sets the type of solver to be used. 
		 * TODO right now there is only one solver (sgdsolver) that does not do anything drastic
		 */
		void set_solver_type(string s);

		/**
		 * Will return an appropriate solver based on @sp and @backend (@backend is prioritized)
		 * vector<DataPackage> set is taken into consideration but should not be taken into consideration, it is checked at
		 * a later stage. TODO remove data set dependency
		 */
		Solver* pick_solver(SolverParameter sp, const string& backend);

		enum class Solvers
		{
			NOT_SUPPORTED,
			SGD //stochastic gradient descent
		};

		Proto_io _io;
		//enums
		Solvers _solver_type;

		SolverFactory(const SolverFactory&) = delete;
};
#endif
