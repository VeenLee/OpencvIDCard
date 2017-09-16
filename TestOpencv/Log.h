#pragma once
#include "spdlog\spdlog.h"

extern std::shared_ptr<spdlog::logger> console;

namespace util
{
	//template<class T> void log(const std::string tag, const T & t);
	template<class T> void log(const std::string tag, const T & t);
	template<class T> void log(const T & t);
}


//#define log(str) (console->info(str));
template<class T>
inline void util::log(const std::string tag, const T & t) {
	console->info(tag);
	console->info(t);
}

template<class T>
void util::log(const T & t) {
	console->info(t);
}


