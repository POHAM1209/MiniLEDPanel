#include"Test.h"
namespace PZTIMAGE {

	myTimer::myTimer() {
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBeginTime);
	}

	myTimer::~myTimer() {
		QueryPerformanceCounter(&nEndTime);
		double time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart * 1000;
		std::cout << "The used time is  " << time << "  ms\n";
	}

	bool TestTest() {
		bool res = false;
		
        //Test Time
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录程序开始时间  
        //Code Start
        

        //Code end
        auto end_time = std::chrono::high_resolution_clock::now(); // 记录程序结束时间  
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); // 计算程序运行时间（以微秒为单位）  
        //std::cout << std::endl << std::endl << "程序运行时间:" << duration / 1000000 << "秒";// << std::endl;
		return res;
	}

}
