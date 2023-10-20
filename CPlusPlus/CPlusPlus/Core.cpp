#include "Core.h"

namespace PZTIMAGE {

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTImage
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTImage::PZTImage(){
		
	}
	
	PZTImage::PZTImage(const std::string& t_fileName){
		m_container = cv::imread(t_fileName, cv::IMREAD_ANYCOLOR);
		m_mark = cv::Mat(m_container.rows, m_container.cols, CV_8UC3, cv::Scalar(255));
	}

	PZTImage::PZTImage(const PZTImage& t_other){
		m_container = t_other.m_container;
		m_mark = t_other.m_mark;
	}

	PZTImage& PZTImage::operator=(const PZTImage& t_other){
		m_container = t_other.m_container;
		m_mark = t_other.m_mark;
		return *this;
	}

	PZTImage::~PZTImage(){
	
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTRegions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTRegions::PZTRegions() {

	}

	PZTRegions::PZTRegions(cv::InputArray t_regI){
		m_container = t_regI.getMat();
		
		// confirm mat::type == CV_16U

		_UpdataRegionNum();
		_UpdataRegionFeatures();
	}

	PZTRegions::PZTRegions(const PZTRegions& t_other) {

	}

	PZTRegions& PZTRegions::operator= (const PZTRegions& t_other) {

		return *this;
	}

	PZTRegions::~PZTRegions() {

	}

	RegionFeature PZTRegions::GetRegionFeature(unsigned int t_index) {
		if (t_index > m_regNum)
			return RegionFeature();
		else
			return m_features[t_index];
	}

	bool PZTRegions::Connection() {
		if (m_regNum != 1)
			return false;

		cv::connectedComponents(m_container, m_container, 8, CV_16U);
		_UpdataRegionNum();
		_UpdataRegionFeatures();
	}

	bool PZTRegions::_UpdataRegionNum(){
		double maxValue = 0;
		cv::minMaxLoc(m_container, &maxValue, nullptr, nullptr, nullptr);
		m_regNum = maxValue;
		
		return true;
	}

	bool PZTRegions::_UpdataRegionFeatures(){
		m_features.clear();
		m_features.reserve(m_regNum);
		
		//....

		return true;
	}
	
	bool TestCore() {
		bool res = false;

		return res;
	}

}
