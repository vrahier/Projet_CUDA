#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

#include "cpu_timer.h"

int NB_PIGEONS = 10;
int NB_PIGEONNIERS = 10;

void satisfy_cpu(int min, int max, bool * satisfy, int * matrice, int CMAX, int LMAX){

	//Pour chaque ligne, on calcule le nombre de 1 présents
	for(int i=0;i<LMAX;i++){

		int somme=0;

		for(int j=0;j<CMAX;j++){
			somme+= matrice[CMAX*i+j];

		}

		//Si la somme est comprise entre min et max alors la contrainte est respectée sur la ligne
		satisfy[i] = ((somme>=min)&&(somme<=max));
	}
}

//Calcule la transposée d'une matrice (permet de passer de la matrice ligne pigeons colonne pigeonniers à la matrice
//ligne pigeonniers colonne pigeons)
void transposeMatriceTab(int * m, int * mInverse,int CMAX, int LMAX){

	for(int j=0;j<CMAX;j++){
		for(int i=0;i<LMAX;i++){

			mInverse[j * LMAX + i] = m[i*CMAX+j];

		}
	}
}


//Pour le moment la matrice contient des 0 partout sauf dans le cas où i=j
void remplirMatriceTab(int * matrice, int CMAX, int LMAX){

	for(int i=0;i<LMAX;i++){

		for(int j=0;j<CMAX;j++){

			matrice[ i * CMAX + j] = i == j ? 1 : 0;

		}
	}
}

void afficherMatriceTab(int * matrice, int CMAX, int LMAX){

	for(int i=0;i<CMAX*LMAX;i++){

		std::cout << matrice[i];

		//Si i+1 est un multiple du nombre de colonne alors on retourne à la ligne
		if(((i+1)%CMAX)==0){
			std::cout << endl;
		}

	}

}


int main(int argc, char ** argv){

	if((argv[1])&& (argv[2])){
		NB_PIGEONS = atoi(argv[1]);
		NB_PIGEONNIERS = atoi(argv[2]);
	}else
	{
		cout<<"veuillez remplir en paramètre le nombre de pigeon et de pigeonniez de cette façon :"<<endl<<"./satifsy.exe NB_PIGEONS NB_PIGONNIERS"<<endl;
		exit (EXIT_FAILURE);
	}


	int * cpu_matrice_tab = new int [NB_PIGEONS*NB_PIGEONNIERS];
	int * cpu_matriceTrans = new int[NB_PIGEONS*NB_PIGEONNIERS];

	bool * cpu_satisfy_pigeons = new bool[NB_PIGEONS];
	bool * cpu_satisfy_pigeonniers = new bool[NB_PIGEONNIERS];

	remplirMatriceTab(cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);


	cout << "Matrice" << endl;
//	afficherMatriceTab(cpu_matrice_tab,pigeonniers,pigeons);
	cout << endl;

	transposeMatriceTab(cpu_matrice_tab,cpu_matriceTrans,NB_PIGEONNIERS,NB_PIGEONS);

	cout << "Matrice transposée" << endl;
//	afficherMatriceTab(cpu_matriceTrans,pigeons,pigeonniers);
	cout << endl;

	CPUTimer c_timer;
	c_timer.start();
	satisfy_cpu(1,1,cpu_satisfy_pigeons,cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);

	satisfy_cpu(0,1,cpu_satisfy_pigeonniers,cpu_matriceTrans,NB_PIGEONS,NB_PIGEONNIERS);


	c_timer.stop();

/*	std::cout << "Satisfy pigeons" << std::endl;
	for(int i=0;i<pigeons;i++){

		std::cout << cpu_satisfy_pigeons[i] << " ";

	}



	std::cout << std::endl;
	std::cout << std::endl;
*/
	std::cout << "Satisfy pigeonniers" << std::endl;

/*	for(int i=0;i<pigeonniers;i++){

		std::cout << cpu_satisfy_pigeonniers[i] << " ";

	}

	std::cout << std::endl;
	std::cout << std::endl;
*/
	cout << "cpu time(ms)=" << c_timer << endl;

	return 0;
}
