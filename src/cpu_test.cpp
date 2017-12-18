/*
 * cpu_test.cpp
 *
 *  Created on: 28 nov. 2017
 *      Author: Paulin VIOLETTE
 */
#include <iostream>
#include <set>
#include <stack>
#include <sstream>
#include "Utilities/file.h"
#include "Solver.h"
#include "cpu_timer.h"


int main(int argc, char * argv[]){

    int p = std::atoi(argv[1]);
    int q = std::atoi(argv[2]);

    std::stringstream ss;
    ss << "pb_pigeons_" << p << "_" << q << ".txt";

    std::string file_name = ss.str();

    Solver solver;
    solver.parseFile(file_name.c_str());

    solver.display_full(std::cout);
    solver.display_readable_formula(std::cout);

    std::set<Solver::var *> unasigned_vars;
    for(auto it = solver._var_table.begin(); it != solver._var_table.end(); ++it){
        unasigned_vars.insert(it->second);
    }

    solver.display_full(std::cout);

    std::srand(std::time(NULL));

    std::cout << "Starting" << std::endl;
    CPUTimer timer;
    timer.start();
    bool hasSol = solver.solve();
    timer.stop();
    if(hasSol){
        solver.display_var_value(std::cout);
        std::cout << "\033[32mSolution trouvé à" << file_name << "\033[0m" << std::endl; //TODO : feu d'artifice
    }
    else{
        std::cout << "\033[31mPas de solution à "<< file_name << "\033[0m" << std::endl;
    }

    timer.print(std::cout);
	return 0;
}
