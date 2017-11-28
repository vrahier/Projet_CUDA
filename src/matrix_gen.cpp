/**
 * Paulin VIOLETTE
 */
#include "matrix_gen.h"
#include <iostream>
/*
 * L'ensemble des matrices possible est toujours parcouru dans le même ordre.
 * On commence par la matrice ne contenant que des 0,
 * Pour passer à la matrice suivante, on ajoute 1 à la matrice en la traitant comme un nombre binaire
 *
 * Exemple :
 * Matrice
 *  1 1 0
 *  1 0 0
 *  0 1 1
 *  Representation en ligne : 110100011
 *  On ajoute 1 au nombre à l'envers (on commence à l'indice 0)
 *  110001011 + 1 = 110001100
 *  Matrice suivante :
 *  0 0 1
 *  1 0 0
 *  0 1 1
 *
 *  On parcours ainsi toute les matrices possibles.
 *  Exemple de parcours pour une matrice de taille 1*4
 *
 *  -> 0000
 *  -> 1000
 *  -> 0100
 *  -> 1100
 *  -> 0010
 *  -> 1010
 *  -> 0110
 *  -> 1110
 *  -> 0001
 *  -> 1001
 *  -> 0101
 *  -> 1101
 *  -> 0011
 *  -> 1011
 *  -> 1111
 */

void init_first_matrix(char * matrix, long int size){
	for(long int i = 0; i < size; ++i){
		matrix[i] = 0;
	}
}


/**
 * @param matrix la matrice à modifier
 * @param size la taille de la matrice
 * @return 1 si la matrice est rempli de 1
 */
int next_matrix(char * matrix, long int size){
	int toAdd = 1;
	long int i = 0;
	while(toAdd != 0 && i < size){
		std::cout << (int)matrix[i] << " -> ";
		matrix[i] = (matrix[i] + 1) % 2;
		std::cout << (int)matrix[i] << " r:";
		toAdd = matrix[i] == 0;
		std::cout << toAdd << std::endl;
		i++;
	}

	return toAdd == 1;
}
