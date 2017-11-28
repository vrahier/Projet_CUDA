//
// Created by etudiant on 28/11/17.
//

#include "Formule.h"

#include <ostream>
#include <iterator>
#include <sstream>
#include <vector>
#include <iostream>

Formule::Formule(int m_min, int m_max) : m_min(m_min), m_max(m_max) {}

Formule Formule::parseString(std::string str) {

    /*
     * Séparation de la chaine en mot
     */
    std::cout << "Formule::parseString \"" << str << "\"" << std::endl;
    std::istringstream iss(str);

    std::vector<std::string> tokens;

    std::copy(std::istream_iterator<std::string>(iss),
              std::istream_iterator<std::string>(),
              std::back_inserter(tokens));

    /*
     * Premier et deuxieme mot : min et max
     */

    int min = std::atoi(tokens[0].c_str());
    int max = std::atoi(tokens[1].c_str());

    Formule f(min, max);

    /*
     * Reste des mots : atomes. Appelle de la methode parseString de Atome pour chacun des mots.
     */
    for(int i = 2; i < tokens.size(); ++i){
        std::cout << "\tFormule::parseString token[" << i << "] = " << tokens[i] << std::endl;
        f.addAtome(Atome::parseString(tokens[i]));
    }
    return f;
}

void Formule::addAtome(const Atome a) {
    m_atomes.insert(a);
}

const std::set<Atome> &Formule::atomes() const {
    return m_atomes;
}

const int Formule::getMin() const{
    return m_min;
}

const int Formule::getMax() const {
    return m_max;
}

std::ostream &operator<<(std::ostream &out, const Formule &f) {

    out << f.getMin() <<  " " << f.getMax() << " ";

    for(std::set<Atome>::const_iterator it = f.atomes().begin(); it != f.atomes().end(); ++it ){
        out << *it << " ";
    }

    return out;
}
