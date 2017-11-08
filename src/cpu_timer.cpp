#include "cpu_timer.h"


CPUTimer::CPUTimer() {
	m_event_stop = m_event_start = std::chrono::high_resolution_clock::now();
}

void CPUTimer::start() {
	m_event_start = std::chrono::high_resolution_clock::now();
}

void CPUTimer::stop() {
	m_event_stop = std::chrono::high_resolution_clock::now();
}

ostream& CPUTimer::print(ostream& out) {
	auto ms = m_event_stop - m_event_start;
	std::chrono::hours   hh = std::chrono::duration_cast<std::chrono::hours>(ms);
	std::chrono::minutes mm = std::chrono::duration_cast<std::chrono::minutes>(ms % chrono::hours(1));
	std::chrono::seconds ss = std::chrono::duration_cast<std::chrono::seconds>(ms % chrono::minutes(1));
	std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(ms % chrono::seconds(1));
	out << setfill('0') << setw(2) << hh.count() << "::"
			<< setw(2) << mm.count() << "::"
			<< setw(2) << ss.count() << "::"
			<< setw(3) << msec.count();

	double total_in_ms = (ss.count() + 60 * mm.count() + 3600 * hh.count()) * 1000 + msec.count();
	out << "|" << total_in_ms;
	double total_in_s = (ss.count() + 60 * mm.count() + 3600 * hh.count()) + (msec.count() / 1000.0);
	out << "|" << total_in_s;
	return out;
}

long int CPUTimer::get_milli_seconds() {
	auto diff = m_event_stop - m_event_start;
	std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
	return static_cast<long int>(msec.count());
}
