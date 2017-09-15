#pragma once
#include "spdlog\spdlog.h"

auto console = spdlog::stdout_color_mt("log");

namespace util
{
	template<class T> void log(T & t);
}


//#define log(str) (console->info(str));
template<class T>
void util::log(T & t) {
	console->info(t);
}


