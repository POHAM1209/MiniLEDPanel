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

	PZTImage& PZTImage::operator = (PZTImage&& t_other){
		m_image = t_other.m_image;
		m_mask = t_other.m_mask;
		return *this;
	}

	PZTImage::~PZTImage(){
	
	}

	bool PZTImage::Empty() const{
		return m_image.empty();
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
		m_regions(),
		m_featuresPtr(nullptr),
		m_regionNum(0),
		m_isRegionChanged(false)
	{
		
	}

	PZTRegions::PZTRegions(cv::Mat t_reg) :
		m_featuresPtr(nullptr),
		m_isRegionChanged(false)
	{
		// confirm t_rg
		if(t_reg.type() != CV_8UC1)
			return;

		m_regions = t_reg.clone();
		_UpdataRegionNum();
	}

	PZTRegions::PZTRegions(const PZTRegions& t_reg, const std::vector<uint32_t>& t_indexs) : 
		m_regions(),
		m_featuresPtr(nullptr),
		m_regionNum(0),
		m_isRegionChanged(false)
	{
		// confirm t_reg and t_indexs
		if(t_reg.Empty())
			return;

		uint32_t regNum = t_reg.GetRegionNum();
		uint32_t maxIdx = *(std::max_element(t_indexs.begin(), t_indexs.end()));
		if(maxIdx > regNum)
			return;

		// updata m_regionNum
		m_regionNum = t_indexs.size();

		// updata m_regions and m_featuresPtr
		cv::Mat tmp;
		uint32_t index = 0, idx = 0;
		m_regions = cv::Mat::zeros(t_reg.m_regions.rows, t_reg.m_regions.cols, CV_8UC1);
		if(t_reg.m_featuresPtr != nullptr){

			m_featuresPtr = std::make_shared<std::vector<RegionFeature>>();
			m_featuresPtr->reserve(m_regionNum);

			for(idx = 0; idx < m_regionNum; ++idx){
				index = t_indexs[idx];
				m_featuresPtr->push_back( (*t_reg.m_featuresPtr)[index] );
				cv::inRange(t_reg.m_regions, cv::Scalar(index + 1), cv::Scalar(index + 1), tmp);
				tmp = tmp - 255 + index + 1;		
				m_regions += tmp;
			}

		}else{

			for(idx = 0; idx < m_regionNum; ++idx){
				index = t_indexs[idx];
				cv::inRange(t_reg.m_regions, cv::Scalar(index + 1), cv::Scalar(index + 1), tmp);
				tmp = tmp - 255 + index + 1;
				m_regions += tmp;
			}
		}

	}

	PZTRegions::PZTRegions(const PZTRegions& t_other){
		m_regions = t_other.m_regions.clone();

		if(t_other.m_featuresPtr == nullptr)
			m_featuresPtr = nullptr;
		else
			m_featuresPtr = t_other.m_featuresPtr;

		m_regionNum = t_other.m_regionNum;
		m_isRegionChanged = false;
	}

	PZTRegions::PZTRegions(PZTRegions&& t_other){
		m_regions = t_other.m_regions;

		if(t_other.m_featuresPtr == nullptr)
			m_featuresPtr = nullptr;
		else
			m_featuresPtr = t_other.m_featuresPtr;
		
		m_regionNum = t_other.m_regionNum;
		m_isRegionChanged = false;
	}

	PZTRegions& PZTRegions::operator= (const PZTRegions& t_other) {
		m_regions = t_other.m_regions.clone();

		if(t_other.m_featuresPtr == nullptr)
			m_featuresPtr = nullptr;
		else
			m_featuresPtr = t_other.m_featuresPtr;
		
		m_regionNum = t_other.m_regionNum;
		m_isRegionChanged = false;

		return *this;
	}

	PZTRegions::~PZTRegions() {

	}

	bool PZTRegions::Empty() const{
		return m_regions.empty();
	}

	RegionFeature PZTRegions::GetRegionFeature(unsigned int t_index){
		// confirm t_index
		if(t_index > m_regionNum)
			return RegionFeature();

		// That m_featuresPtr is nullptr represents no objects in the container of feature, and 
		// that m_isRegionChanged is true represents that the shared pointer(m_featuresPtr) is invalid.
		if(m_featuresPtr == nullptr || m_isRegionChanged){
			m_featuresPtr = std::make_shared<std::vector<PZTIMAGE::RegionFeature>>(); 
			m_featuresPtr->reserve(m_regionNum);
		}

		// Whether updata the container of feature
		if(m_featuresPtr->size() == 0 || m_isRegionChanged){
			if(!_UpdataRegionFeatures())
				return RegionFeature();
		}
		m_isRegionChanged = false;

		return (*m_featuresPtr)[t_index];		
	}

	double PZTRegions::GetRegionFeature(unsigned int t_index, FeatureType t_type){
		// confirm t_index
		if(t_index > m_regionNum)
			return 0;
		
		if(m_featuresPtr == nullptr || m_isRegionChanged){
			m_featuresPtr = std::make_shared<std::vector<PZTIMAGE::RegionFeature>>();
			m_featuresPtr->reserve(m_regionNum);
		}
			

		if(m_featuresPtr->size() == 0 || m_isRegionChanged){
			if(!_UpdataRegionFeatures())
				return 0;
		}
		m_isRegionChanged = false;

		RegionFeature value = (*m_featuresPtr)[t_index];
		double res;
		switch(t_type){
			case FEATURETYPE_AREA: 			res = value.m_area; break;
			case FEATURETYPE_CIRCULARITY: 	res = value.m_circularity; break;
			case FEATURETYPE_ROW: 			res = value.m_row; break;
			case FEATURETYPE_COL: 			res = value.m_col; break;
		default:
			res = 0; break;
		}

		return res;
	}

	bool PZTRegions::Connection() {
		if(m_regionNum == 0)
			return true;

		// confirm m_regions
		if(m_regions.empty())
			return false;

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
			m_isRegionChanged = true;
			return true;
		}
	}

	bool PZTRegions::Disconnection(){
		if(m_regionNum == 1 || m_regionNum == 0)
			return true;

		// confirm m_regions
		if(m_regions.empty())
			return false;
		
		cv::threshold(m_regions, m_regions, 0, 1, cv::THRESH_BINARY);
		m_regionNum = 1;
		m_isRegionChanged = true;
			
		return true;
	}

	bool PZTRegions::FillUp() {
		if(m_regionNum == 0)
			return true;

		// confirm m_regions and m_regionNum
		if(m_regions.empty() || m_regionNum != 1)
			return false;
		
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(m_regions, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE); // gain external contours
		cv::drawContours(m_regions, contours, -1, cv::Scalar(1), cv::FILLED);

		m_isRegionChanged = true;

		return true;
	}

	bool PZTRegions::Erosion(StructElement t_elm, unsigned int t_kernelLen){
		if(m_regionNum == 0)
			return true;

		// confirm m_regions and m_regionNum
		if(m_regions.empty() || m_regionNum != 1)
			return false;

		unsigned int kernelLen = t_kernelLen | 0x00000001;
		cv::Mat kernel = cv::getStructuringElement(static_cast<cv::MorphShapes>(t_elm), cv::Size(kernelLen, kernelLen));
		cv::erode(m_regions, m_regions, kernel);

		m_isRegionChanged = true;

		return true;
	}

	unsigned int PZTRegions::GetRegionNum() const{
		return m_regionNum;
	}

	bool PZTRegions::_UpdataRegionNum(){
		double maxValue = 0;
		cv::minMaxLoc(m_regions, nullptr, &maxValue, nullptr, nullptr);
		m_regionNum = maxValue;
		
		return true;
	}

	void PZTRegions::DisplayRegion(){
		cv::imshow("m_regions", m_regions);
		cv::waitKey(0);
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

	bool PZTRegions::_UpdataRegionsFeaturesV2(){
		m_featuresPtr->clear();
		m_featuresPtr->reserve(m_regionNum);

		// ...
		cv::Mat oneRegion;
		RegionFeature trait;
		for(unsigned int idx = 1; idx <= m_regionNum; ++idx){
			// 获得每个连通域
			oneRegion = m_regions - idx;
			cv::threshold(oneRegion, oneRegion, 0, 1, cv::THRESH_BINARY_INV);

			// _UpdataOneRegionFeatures
			trait = _GainOneRegionFeatures(oneRegion);

			m_featuresPtr->push_back(trait);
		}

		return true;
	}

	RegionFeature PZTRegions::_GainOneRegionFeatures(cv::InputArray t_oneRegion){
		cv::Mat oneRegion = t_oneRegion.getMat();

		RegionFeature trait;

		cv::Moments moms = cv::moments(t_oneRegion);

		// features--area
		// trait.m_area = (cv::sum(oneRegion))[0];
		trait.m_area = moms.m00;

		// features--col
		trait.m_col = moms.m10 / moms.m00;

		// features--row
		trait.m_row = moms.m01 / moms.m00;

		return RegionFeature();
	}

	bool PZTRegions::_UpdataRegions(){
		//cv::threshold(m_regions, )

		return false;
	}


	
	bool TestCore() {
		bool res = false;

		// 测试 PZTRegion 构造函数

		cv::Mat reg = cv::imread("./connectedDomain.jpg");
		cv::cvtColor(reg, reg, cv::COLOR_RGB2GRAY);

		cv::threshold(reg, reg, 100, 1, cv::THRESH_BINARY);

		PZTRegions obj(reg);
		//obj.DisplayRegion();

		obj.Connection();
		//obj.DisplayRegion();

		RegionFeature feature = obj.GetRegionFeature(0);


		PZTRegions obj1(obj, std::vector<uint32_t>{0, 2});

		return res;
	}

}
