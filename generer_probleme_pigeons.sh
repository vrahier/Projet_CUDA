#!/bin/bash

#Ce script génère un fichier interprétable par le programme de résolution pour le problème des pigeons et des pigeonniers
#Il prend comme paramètres N le nombre de pigeons et Q le nombre de pigeonniers

N=$1
Q=$2


#On crée le fichier de sortie
nomFichierSortie="pb_pigeons_"$N"_"$Q".txt";
touch $nomFichierSortie;

#On vide le fichier s'il existe déjà
>$nomFichierSortie;

#On génère les contraintes sur les pigeons
for i in `seq 1 $N`;
do
    ligne="1 1";
    
    for j in `seq 1 $Q`;
    do
        ligne=$ligne" X"$j"_"$i;
    done

    echo $ligne >> $nomFichierSortie;
done

#On génère les contraintes sur les pigeonniers
for i in `seq 1 $Q`;
do
    ligne="0 1";
    
    for j in `seq 1 $N`;
    do
        ligne=$ligne" X"$i"_"$j;
    done

    echo $ligne >> $nomFichierSortie; 

done
