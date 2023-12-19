#ifndef _TEST_H_
#define _TEST_H_

#include <chrono>  
#include <iostream> 
#include <string>
#include<Windows.h>
// ²âÊÔ

namespace PZTIMAGE {

	class myTimer {
	public:
		myTimer();
		~myTimer();
	private:
		LARGE_INTEGER nFreq;
		LARGE_INTEGER nBeginTime;
		LARGE_INTEGER nEndTime;
	};

	bool TestTest();

}

#endif
