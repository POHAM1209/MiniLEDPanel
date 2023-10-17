#ifndef _CORE_H_
#define _CORE_H_

#include <opencv2/core.h>

namespace PZTIMAGE{

	class PZTImage{
	private:
		cv::Mat m_container;
	};

	bool TestCore();

}

#endif
