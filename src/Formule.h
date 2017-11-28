//
// Created by pviolette on 28/11/17.
//

#ifndef PROJET_CUDA_FORMULE_H
#define PROJET_CUDA_FORMULE_H

#include <set>
#include "Atome.h"

class Formule {

private:
    int m_min;
    int m_max;
    std::set<Atome> m_atomes;

public:

    static Formule parseString(std::string str);

    Formule(int m_min, int m_max);

    Formule(std::string str);

    void addAtome(const Atome a);

    const std::set<Atome>& atomes() const;

    const int getMin() const;
    const int getMax() const;


};

std::ostream& operator<<(std::ostream& out, const Formule& f);


#endif //PROJET_CUDA_FORMULE_H
