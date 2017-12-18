#include "cume_base.h"

using namespace cume;

Exception::Exception() : std::exception() { }

Exception::Exception(string m, const char *f, int l) : std::exception() {
	message = m;
	file = f;
	line = l;
}

Exception::Exception(string m, string a, const char *f, int l)  : std::exception() {
	message = m;
	action = a;
	file = f;
	line = l;
}

Exception::Exception(string m, string a, string s, const char *f, int l)  : std::exception() {
	message = m;
	action = a;
	solution = s;
	file = f;
	line = l;
}

Exception::~Exception() throw() {
}

const char* Exception::what() const throw() {
	ostringstream oss;
	oss << "CUME Exception: " << message << endl;
	oss << "\tin " << file << " at line " << line << endl;
	if (action.size() != 0) {
		oss << "action: " << action << endl;
	}
	if (solution.size() != 0) {
		oss << "solution: " << solution << endl;
	}
	string str = oss.str();
	return str.c_str();
}


