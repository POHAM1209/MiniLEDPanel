#include "Core.h"

namespace PZTIMAGE {

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTImage
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTImage::PZTImage(){
		
	}
	
	PZTImage::PZTImage(const std::string& t_fileName){
		m_image= cv::imread(t_fileName, cv::IMREAD_ANYCOLOR);
		m_mask = cv::Mat(m_image.rows, m_image.cols, CV_8UC1, cv::Scalar(1));
	}

	PZTImage::PZTImage(const PZTImage& t_other){
		m_image = t_other.m_image;
		m_mask = t_other.m_mask.clone();
	}

	PZTImage& PZTImage::operator=(const PZTImage& t_other){
		m_image = t_other.m_image;
		m_mask = t_other.m_mask.clone();
		return *this;
	}

	PZTImage::~PZTImage(){
	
	}

	bool PZTImage::GetImageSize(unsigned int& t_imgRow, unsigned int& t_imgCol){
		bool res = false;

		return res;
	}

	bool PZTImage::Compose(const cv::Mat& t_ch0, const cv::Mat& t_ch1, const cv::Mat& t_ch2){
		bool res = false;

		return res;
	}

	bool PZTImage::Decompose(cv::Mat& t_ch0, cv::Mat& t_ch1, cv::Mat& t_ch2){
		bool res = false;

		return res;
	}

	bool PZTImage::ReduceDomain(const PZTRegions& t_reg){
		bool res = false;

		return res;
	}

	bool PZTImage::RGB2Gray(){
		bool res = false;

		return res;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTRegions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTRegions::PZTRegions():
		m_regions(),
		m_regionNum(0),
		m_featuresPtr()
	{
		
	}

	PZTRegions::PZTRegions(cv::Mat t_reg) :
		m_regions(),
		m_regionNum(0)
	{
		_UpdataRegionNum();
		m_featuresPtr = std::make_shared<std::vector<RegionFeature>>(m_regionNum);
	}

	PZTRegions::PZTRegions(const PZTRegions& t_other) {
		m_regions = t_other.m_regions;
		m_featuresPtr = t_other.m_featuresPtr;
		m_regionNum = t_other.m_regionNum;
	}

	PZTRegions& PZTRegions::operator= (const PZTRegions& t_other) {
		m_regions = t_other.m_regions;
		m_featuresPtr = t_other.m_featuresPtr;
		m_regionNum = t_other.m_regionNum;

		return *this;
	}

	PZTRegions::~PZTRegions() {

	}

	RegionFeature PZTRegions::GetRegionFeature(unsigned int t_index) {
		// confirm
		if(t_index > m_regionNum)
			return RegionFeature();

		int size = m_featuresPtr->size();
		if(size == 0 || size != m_regionNum)
			bool res = _UpdataRegionFeatures();

		return (*m_featuresPtr)[t_index];
	}

	bool PZTRegions::Connection() {
		// confirm
		if(m_regions.data == nullptr)
			return false;

		cv::Mat tmp16, tmp8;
		cv::threshold(m_regions, tmp16, 0, 255, cv::THRESH_BINARY);
		tmp16.convertTo(tmp8, CV_8U);
		m_regionNum = cv::connectedComponents(tmp8, m_regions, 8, CV_16U);
	
		return true;
	}

	bool PZTRegions::Disconnection(){
		// confirm
		if(m_regions.data == nullptr)
			return false;
		
		m_regionNum = 1;
		cv::threshold(m_regions, m_regions, 0, 1, cv::THRESH_BINARY);

		return true;
	}

	bool PZTRegions::Erosion(StructElement t_elm){
		
		return true;
	}

	bool PZTRegions::_UpdataRegionNum(){
		double maxValue = 0;
		cv::minMaxLoc(m_regions, &maxValue, nullptr, nullptr, nullptr);
		m_regionNum = maxValue;
		
		return true;
	}

	bool PZTRegions::_UpdataRegionFeatures(){
		m_featuresPtr->clear();
		m_featuresPtr->reserve(m_regionNum);
		
		//....
		std::vector<std::vector<cv::Point>> contours;
		cv::Mat tmp16, tmp8;
		cv::threshold(m_regions, tmp16, 0, 255, cv::THRESH_BINARY);
		tmp16.convertTo(tmp8, CV_8U);
		//cv::findContours(m_regions, contours, )
		// 做findCoontours的实验
		
		return true;
	}

	bool PZTRegions::_UpdataRegions(){
		//cv::threshold(m_regions, )

		return false;
	}
	
	bool TestCore() {
		bool res = false;

		cv::Mat img = cv::imread(std::string("E:/gray.png"), cv::IMREAD_GRAYSCALE);
		
		std::cout << "   ---    " << (int)(img.data[60]) << "\n";


		auto imgCpy = img;
		cv::pow(imgCpy, 2, imgCpy);

		std::cout << (int)(imgCpy.data[60]) << "   ---    " << (int)(img.data[60]);

		return res;
	}

}
