#ifndef _CORE_H_
#define _CORE_H_

#include <opencv2/core.h>

#include <vector>

namespace PZTIMAGE{

	// define struct
	typedef struct RegionFeature{
		unsigned int 			m_area;
		
	}RegionFeature;


	// define class
	class PZTImage{
	private:
		cv::Mat m_container;
		cv::Mat m_mark;
	};

	class PZTRegions{
	
	private:
		cv::Mat      			m_container;
		unsigned int 			m_regNum;
		std::vector<RegionFeature> 	m_features;
	};

	bool TestCore();

}

#endif
