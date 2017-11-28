#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

#include "cpu_timer.h"

int NB_PIGEONS =0;
int NB_PIGEONNIERS =0;

void satisfy_cpu(int min, int max, bool * satisfy, char * matrice, int CMAX, int LMAX){

	//Pour chaque ligne, on calcule le nombre de 1 présents
	for(long int i=0;i<LMAX;i++){

		unsigned int somme=0;

		for(long int j=0;j<CMAX;j++){
			somme+= matrice[CMAX*i+j];

		}

		//Si la somme est comprise entre min et max alors la contrainte est respectée sur la ligne
		satisfy[i] = ((somme>=min)&&(somme<=max));
	}
}

bool satisfy(bool * satisfy_tab, int size){

	for(int i=0;i<size;i++){

		if(!satisfy_tab[i]){
			return false;
		}

	}

	return true;

}

//Calcule la transposée d'une matrice (permet de passer de la matrice ligne pigeons colonne pigeonniers à la matrice
//ligne pigeonniers colonne pigeons)
void transposeMatriceTab(char * m,char * mInverse,int CMAX, int LMAX){

	for(long int j=0;j<CMAX;j++){
		for(long int i=0;i<LMAX;i++){

			mInverse[j * LMAX + i] = m[i*CMAX+j];

		}
	}
}


//Pour le moment la matrice contient des 0 partout sauf dans le cas où i=j
void remplirMatriceTab(char * matrice, int CMAX, int LMAX){

	for(long int i=0;i<LMAX;i++){

		for(long int j=0;j<CMAX;j++){

			matrice[ i * CMAX + j] = i == j ? 1 : 0;

		}
	}
}



int main(int argc, char ** argv){

	if((argv[1])&& (argv[2])){
		NB_PIGEONS = atoi(argv[1]);
		NB_PIGEONNIERS = atoi(argv[2]);
	}else
	{
		cout<<"veuillez remplir en paramètre le nombre de pigeons et de pigeonniers de cette façon :"<<endl;
		cout<<"./satifsy.exe NB_PIGEONS NB_PIGONNIERS"<<endl;
		exit (EXIT_FAILURE);
	}


	char * cpu_matrice_tab = new char [NB_PIGEONS*NB_PIGEONNIERS];
	char * cpu_matriceTrans = new char[NB_PIGEONS*NB_PIGEONNIERS];

	bool * cpu_satisfy_pigeons = new bool[NB_PIGEONS];
	bool * cpu_satisfy_pigeonniers = new bool[NB_PIGEONNIERS];

	remplirMatriceTab(cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);


	transposeMatriceTab(cpu_matrice_tab,cpu_matriceTrans,NB_PIGEONNIERS,NB_PIGEONS);


	CPUTimer c_timer;
	c_timer.start();
	satisfy_cpu(1,1,cpu_satisfy_pigeons,cpu_matrice_tab,NB_PIGEONNIERS,NB_PIGEONS);

	satisfy_cpu(0,1,cpu_satisfy_pigeonniers,cpu_matriceTrans,NB_PIGEONS,NB_PIGEONNIERS);


	c_timer.stop();


	cout << "cpu time(ms)=" << c_timer << endl;

	return 0;
}
