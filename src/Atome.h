//
// Created by etudiant on 28/11/17.
//

#ifndef PROJET_CUDA_ATOME_H
#define PROJET_CUDA_ATOME_H


#include <string>

class Atome {

private:
    std::string m_varName;
    bool m_negated;

public:

    Atome(const std::string &m_varName, bool m_negated);

    Atome(const std::string &m_varName);

    const std::string& getVarName() const;
    const bool isNegated() const;

    static Atome parseString(std::string str);

    bool operator==(const Atome &rhs) const;

    bool operator!=(const Atome &rhs) const;

    bool operator<(const Atome &rhs) const;

    bool operator>(const Atome &rhs) const;

    bool operator<=(const Atome &rhs) const;

    bool operator>=(const Atome &rhs) const;

};

std::ostream& operator<<(std::ostream& out, const Atome& a);

#endif //PROJET_CUDA_ATOME_H
