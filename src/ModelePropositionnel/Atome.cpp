//
// Created by etudiant on 28/11/17.
//

#include "Atome.h"

#include <ostream>
#include <iostream>

const std::string &Atome::getVarName() const {
    return m_varName;
}

const bool Atome::isNegated() const {
    return m_negated;
}

Atome::Atome(const std::string &m_varName) : m_varName(m_varName), m_negated(false) {}

Atome::Atome(const std::string &m_varName, bool m_negated) : m_varName(m_varName), m_negated(m_negated) {}


bool Atome::operator==(const Atome &rhs) const {
    return m_varName == rhs.m_varName &&
           m_negated == rhs.m_negated;
}

/**
 * Operateur de comparaison inferieur. Compare d'abord sur le nom de la variable, puis ensuite sur la negation ou non de la variable.
 * @param rhs un autre Atome
 * @return la comparaison inferieure sur m_varName, ou, s'il les m_varName sont égaux, la comparason sur m_negated.
 */
bool Atome::operator<(const Atome &rhs) const {
    if (m_varName < rhs.m_varName)
        return true;
    if (rhs.m_varName < m_varName)
        return false;
    return m_negated < rhs.m_negated;
}

/**
 * @see Atome::operator<
 * @param rhs
 * @return
 */
bool Atome::operator>(const Atome &rhs) const {
    return rhs < *this;
}

/**
 * @see Atome::operator<
 * @see Atome::operator==
 * @param rhs
 * @return
 */
bool Atome::operator<=(const Atome &rhs) const {
    return !(rhs < *this);
}

/**
 * @see Atome::operator<
 * @see Atome::operator==
 * @param rhs
 * @return
 */
bool Atome::operator>=(const Atome &rhs) const {
    return !(*this < rhs);
}

/**
 * @see Atome::operator==
 * @param rhs
 * @return
 */
bool Atome::operator!=(const Atome &rhs) const {
    return !(rhs == *this);
}

Atome Atome::parseString(std::string str) {
    std::cout << "\t\tAtome::parseString " << str << std::endl;
    if(str[0] == '-'){
        return Atome(str.substr(1), true);
    }
    else{
        return Atome(str);
    }
}

/**
 * Écris l'atome donné sur le flux de sortie donnée, sous la forme "varName" ou "-varName" selon la valeur de isNegated()
 * @param out un flux de sortie
 * @param a un atome
 * @return out, le flux de sortie donné en paramètre
 */
std::ostream &operator<<(std::ostream &out, const Atome &a) {

    if(a.isNegated()){
        out << "-";
    }
    out << a.getVarName();

    return out;
}

