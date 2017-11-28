/*
 * cpu_test.cpp
 *
 *  Created on: 28 nov. 2017
 *      Author: Paulin VIOLETTE
 */
#include <iostream>
#include "Formule.h"


int main(int argc, char * argv[]){
    Formule f = Formule::parseString("1 2 x1_1 x1_2 x1_3");

	std::cout << f << std::endl;
	return 0;
}
