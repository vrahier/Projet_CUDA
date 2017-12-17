//
// Created by etudiant on 30/11/17.
//

#include <vector>

#include "ModelePropositionnel/Atome.h"
#include "ModelePropositionnel/Formule.h"
#include "ModelePropositionnel/VariableTable.h"

#include "ResolutionStructure.h"
#include "Utilities/file.h"

/**
 * Charge le fichier donnée dans l'ensemble de formule passé en parametre
 * @param ens_formule le vecteur dans lequel seront stockées les formules chargées.
 * @param filePath le chemin du fichier à charger
 */
void loadFile(std::vector<Formule> & ens_formule, VariableTable& varTable const char * filePath){
    File file(filePath);

    ens_formule.empty();

    for(std::vector<std::string>::const_iterator it = file.begin(); it != file.end(); ++it){
        if( *it != "")
            ens_formule.push_back(Formule::parseString(*it));

    }
}

void initSolution()


void applyValue(int index, Solution * s, const std::vector<Formule>& formules, const VariableTable& variableTable){
}

int main(int argc, char * argv){

}
