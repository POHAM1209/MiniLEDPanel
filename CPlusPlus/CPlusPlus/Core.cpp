#include "Core.h"

namespace PZTIMAGE {

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTImage
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTImage::PZTImage(){
		
	}
	
	PZTImage::PZTImage(const std::string& t_fileName){
		m_image = cv::imread(t_fileName, cv::IMREAD_ANYCOLOR);
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

	bool PZTImage::GetImageSize(unsigned int& t_imgRow, unsigned int& t_imgCol) const{
		// confirm m_image
		if(m_image.empty()){
			t_imgCol = 0;
			t_imgRow = 0;
			return false;
		}

		t_imgCol = m_image.cols;
		t_imgRow = m_image.rows;

		return true;
	}

	bool PZTImage::Compose(const cv::Mat& t_ch0, const cv::Mat& t_ch1, const cv::Mat& t_ch2){
		// confirm input parameters
		if(t_ch0.empty() || t_ch1.empty() || t_ch2.empty())
			return false;
		if(t_ch0.type() + t_ch1.type() + t_ch2.type() != 3 * CV_8UC1)
			return false;

		cv::Mat channels[3] = {t_ch0, t_ch1, t_ch2};
		cv::merge(channels, 3, m_image);

		return true;
	}

	bool PZTImage::Decompose(cv::Mat& t_ch0, cv::Mat& t_ch1, cv::Mat& t_ch2){
		// confirm m_image
		if(m_image.empty() || m_image.channels() != 3)
			return false;

		cv::Mat channels[3];
		cv::split(m_image, channels);

		t_ch0 = channels[0];
		t_ch1 = channels[1];
		t_ch2 = channels[2];

		return true;
	}

	bool PZTImage::ReduceDomain(const PZTRegions& t_reg){
		// confirm t_reg
		if(t_reg.m_regions.empty())
			return false;

		cv::threshold(const_cast<PZTRegions&>(t_reg).m_regions, m_mask, 0, 1, cv::THRESH_BINARY);
		return true;
	}

	bool PZTImage::RGB2Gray(){
		// confirm m_image
		if(m_image.channels() != 3)
			return false;

		cv::cvtColor(m_image, m_image, cv::COLOR_BGR2BGRA);
		return true;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// PZTRegions
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	PZTRegions::PZTRegions():
		m_regionNum(0)
	{
		
	}

	PZTRegions::PZTRegions(cv::Mat t_reg) :
		m_regionNum(0)
	{
		// confirm t_rg
		if(t_reg.type() != CV_8UC1)
			return;

		m_regions = t_reg;
		_UpdataRegionNum();
		m_featuresPtr = std::make_shared<std::vector<PZTIMAGE::RegionFeature>>(m_regionNum);
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
		// confirm t_index
		if(t_index > m_regionNum)
			return RegionFeature();

		int size = m_featuresPtr->size();
		if(size == 0 || size != m_regionNum)
			if(!_UpdataRegionFeatures())
				return RegionFeature();

		return (*m_featuresPtr)[t_index];
	}

	bool PZTRegions::Connection() {
		// confirm m_regions
		if(m_regions.empty())
			return false;

		if(m_regionNum == 0)
			return true;

		if(!Disconnection())
			return false;

		cv::Mat tmp16;
		// connectedComponents param1(8 bit) and param2(16 or 32 bit)   type deferent
		auto regionNum = cv::connectedComponents(m_regions, tmp16, 8, CV_16U);
		if(regionNum > 255)
			return false;
		else{
			tmp16.convertTo(m_regions, CV_8UC1);
			m_regionNum = regionNum - 1;
			return true;
		}
	}

	bool PZTRegions::Disconnection(){
		// confirm m_regions
		if(m_regions.empty())
			return false;
		
		cv::threshold(m_regions, m_regions, 0, 1, cv::THRESH_BINARY);

		cv::Scalar value = cv::sum(m_regions);
		if((long long)value[0] == 0) // Exception: maybe overflow.
			m_regionNum = 0;
		else
			m_regionNum = 1;

		return true;
	}

	bool PZTRegions::FillUp() {
		return true;
	}

	bool PZTRegions::Erosion(StructElement t_elm, unsigned int t_kernelLen){
		// confirm m_regionNum
		if(m_regionNum > 1)
			return false;

		cv::Mat kernel = cv::getStructuringElement(static_cast<cv::MorphShapes>(t_elm), cv::Size(t_kernelLen, t_kernelLen));
		cv::erode(m_regions, m_regions, kernel);

		return true;
	}

	unsigned int PZTRegions::GetRegionNum(){
		return m_regionNum;
	}

	bool PZTRegions::_UpdataRegionNum(){
		double maxValue = 0;
		cv::minMaxLoc(m_regions, nullptr, &maxValue, nullptr, nullptr);
		m_regionNum = maxValue;
		
		return true;
	}

	bool PZTRegions::_UpdataRegionFeatures(){
		m_featuresPtr->clear();
		m_featuresPtr->reserve(m_regionNum);
		
		//....
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(m_regions, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE); // 不建立层次结构

		cv::Point2f point;
		RegionFeature trait;
		size_t contourNum = contours.size();
		for(size_t contourIdx = 0; contourIdx < contourNum; ++contourIdx){
			cv::Moments moms = cv::moments(contours[contourIdx]);

			// features--col
			trait.m_col = moms.m10 / moms.m00;

			// features--row
			trait.m_row = moms.m01 / moms.m00;

			// features--area
			trait.m_area = moms.m00;

			// features--circularity
			// formula: c = (4 * Pi * area_region) / (perimeter_region * perimeter_region)
			double perimeter = cv::arcLength(contours[contourIdx], true);
			trait.m_circularity = 4 * CV_PI * trait.m_area / (perimeter * perimeter);

			// features--outer_radius
			cv::minEnclosingCircle(contours[contourIdx], point, trait.m_outerRadius);

			// features--inner_radius
			//..
			
			m_featuresPtr->push_back(trait);
		}

		return true;
	}

	bool PZTRegions::_UpdataRegions(){
		//cv::threshold(m_regions, )

		return false;
	}
	
	bool TestCore() {
		bool res = false;

		PZTImage img("./domain.jpg");
		res = img.RGB2Gray();

		cv::Mat label;
		cv::Mat image = cv::imread("./domain.jpg");
		cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
		cv::threshold(image, label, 100, 1, cv::THRESH_BINARY);

		PZTIMAGE::PZTRegions reg(label);

		reg.Connection();

		auto tmp = reg.GetRegionFeature(0);

		return res;
	}

}
