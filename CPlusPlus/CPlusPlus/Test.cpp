#include"Test.h"

namespace PZTIMAGE {

	bool TestTest() {
		bool res = false;
		//时间测试
		auto start_time = high_resolution_clock::now(); // 记录程序开始时间  
		// 测试代码  
	

		//代码结束
		auto end_time = high_resolution_clock::now(); // 记录程序结束时间  
		double duration = duration_cast<microseconds>(end_time - start_time).count(); // 计算程序运行时间（以微秒为单位）  
		//std::cout << std::endl << std::endl << "程序运行时间：" << duration / 1000000 << " 秒";//<< endl<<endl<<endl;

		return res;
	}

}
