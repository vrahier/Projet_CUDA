//
// Created by Paulin Violette on 30/11/17.
//

#ifndef PROJECT_RESOLUTIONSTRUCTURE_H
#define PROJECT_RESOLUTIONSTRUCTURE_H


/**
 * Représente un ensemble de formule (aplha, beta, { liste de booléen } 
 */
struct Solution{
    int line, column;
    SolutionLine * lines;
};

struct SolutionLine{
    int min; //Alpha
    int max; //Beta
    int * values; //Tableau de int de taille Solution.column
};

#endif //PROJECT_RESOLUTIONSTRUCTURE_H
