/*
 * cpu_test.cpp
 *
 *  Created on: 28 nov. 2017
 *      Author: Paulin VIOLETTE
 */
#include <iostream>
#include <set>
#include <stack>
#include "Utilities/file.h"
#include "Solver.h"
#include "cpu_timer.h"


int main(int argc, char * argv[]){

    Solver solver;
    solver.parseFile("pb_pigeons_6_5.txt");

    solver.display_full(std::cout);
    solver.display_readable_formula(std::cout);

    std::set<Solver::var *> unasigned_vars;
    for(auto it = solver._var_table.begin(); it != solver._var_table.end(); ++it){
        unasigned_vars.insert(it->second);
    }
/*
    std::stack<Solver::var *> step_vars;

    solver.assign_value(solver._var_table.begin()->second, true, unasigned_vars, step_vars );

    solver.display_full(std::cout);


    while(!step_vars.empty()){
        solver.remove_assign(step_vars.top());
        step_vars.pop();
    }

    solver.display_full(std::cout);*/

    std::srand(std::time(NULL));

    std::cout << "Starting" << std::endl;
    CPUTimer timer;
    timer.start();
    bool hasSol = solver.solve();
    timer.stop();
    if(hasSol){
        std::cout << "\033[32mSolution trouvé\033[0m" << std::endl; //TODO : feu d'artifice
        solver.display_var_value(std::cout);
    }
    else{
        std::cout << "\033[31mPas de solution\033[0m" << std::endl;
    }
    std::cout << "Time : " << timer.get_milli_seconds() << "ms";


	return 0;
}
