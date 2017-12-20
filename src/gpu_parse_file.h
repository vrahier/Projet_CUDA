//
// Created by etudiant on 20/12/17.
//

#ifndef PROJECT_GPU_PARSE_FILE_H
#define PROJECT_GPU_PARSE_FILE_H

#include "CuME/cume.h"

void parseFile(const char *filepath, cume::Matrix<int> &dest, int &n_var, int &n_formulas);

#endif //PROJECT_GPU_PARSE_FILE_H
