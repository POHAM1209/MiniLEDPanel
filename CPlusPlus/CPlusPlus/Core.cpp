#include "Core.h"

namespace PZTIMAGE {

	PZTImage::PZTImage(){
		
	}
	
	PZTImage::PZTImage(const std::string& t_fileName){
		m_container = cv::imread(t_fileName, cv::IMREAD_UNCHANGED);

	}

	PZTImage& PZTImage::operator=(const PZTImage& t_other){
		
	}

	bool TestCore() {
		bool res = false;

		return res;
	}

}
