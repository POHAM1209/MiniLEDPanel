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

	PZTImage::PZTImage(const cv::Mat& t_img, const cv::Mat& t_mask) :
		m_image(),
		m_mask()
	{
		// confirm t_img
		if(t_img.empty())
			return;

		m_image = t_img;

		if(t_mask.empty())
			m_mask = cv::Mat(m_image.rows, m_image.cols, CV_8UC1, cv::Scalar(1));
		else
			m_mask = t_mask.clone();
	}

	PZTImage::PZTImage(const cv::Mat& t_img, cv::Mat&& t_mask){
		// confirm t_img
		if(t_img.empty())
			return;

		m_image = t_img;

		if(t_mask.empty())
			m_mask = cv::Mat(m_image.rows, m_image.cols, CV_8UC1, cv::Scalar(1));
		else
			m_mask = t_mask;
	}

	PZTImage PZTImage::operator - (const PZTImage& t_other) const {
		// confirm self and t_other
		if(t_other.Empty())
			return PZTImage(*this);

		if(this->Empty())
			return PZTImage();
		
		// m_mask
		cv::Mat tmpMask = t_other.m_mask + m_mask;
		cv::inRange(tmpMask, tmpMask, cv::Scalar(2), cv::Scalar(2));

		// m_image
		cv::Mat tmpImg = m_image - t_other.m_image;

		return PZTImage(std::move(tmpImg), tmpMask - 1);
	}

	PZTImage PZTImage::operator + (const PZTImage& t_other) const{
		// confirm self and t_other
		if(t_other.Empty())
			return PZTImage(*this);

		if(this->Empty())
			return PZTImage();

		// m_mask
		cv::Mat tmpMask = t_other.m_mask + m_mask;
		cv::inRange(tmpMask, tmpMask, cv::Scalar(2), cv::Scalar(2));

		// m_image
		cv::Mat tmpImg = m_image + t_other.m_image;

		return PZTImage(std::move(tmpImg), tmpMask - 1);
	}

	PZTImage PZTImage::operator - (uint8_t t_val) const{
		// confirm self 
		if(this->Empty())
			return PZTImage();

		return PZTImage(m_image - t_val, m_mask);
	}

	PZTImage PZTImage::operator + (uint8_t t_val) const{
		// confirm self 
		if(this->Empty())
			return PZTImage();

		return PZTImage(m_image + t_val, m_mask);
	}

	PZTImage::PZTImage(const PZTImage& t_other){
		m_image = t_other.m_image;
		m_mask = t_other.m_mask.clone();
	}

	PZTImage::PZTImage(PZTImage&& t_other){
		m_image = t_other.m_image;
		m_mask = t_other.m_mask;
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

	PZTImage PZTImage::Clone() const{
		return PZTImage(this->m_image.clone(), this->m_mask);
	}

	int PZTImage::Channels() const{
		if(m_image.empty())
			return 0;
		
		return m_image.channels();
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

	bool PZTImage::GetImageSize(cv::Size2i& t_size) const{
		if(m_image.empty()){
			t_size = cv::Size2i(0, 0);
			return false;
		}

		t_size = cv::Size2i(m_image.cols, m_image.rows);

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

	bool PZTImage::Decompose(cv::Mat& t_ch0, cv::Mat& t_ch1, cv::Mat& t_ch2) const{
		// confirm m_image
		if(m_image.empty() || m_image.channels() != 3)
			return false;

		cv::Mat channels[3];
		cv::split(m_image, channels);

		t_ch0 = std::move( channels[0] );
		t_ch1 = std::move( channels[1] );
		t_ch2 = std::move( channels[2] );

		return true;
	}

	bool PZTImage::Decompose(PZTImage& t_ch0, PZTImage& t_ch1, PZTImage& t_ch2) const{
		// confirm m_image
		if(m_image.empty() || m_image.channels() != 3)
			return false;

		cv::Mat channels[3];
		cv::split(m_image, channels);

		t_ch0 = std::move( PZTImage(channels[0], m_mask) );
		t_ch1 = std::move( PZTImage(channels[1], m_mask) );
		t_ch2 = std::move( PZTImage(channels[2], m_mask) );

		return true;
	}

	bool PZTImage::Mean(uint32_t t_maskWidth, uint32_t t_maskHeight){
		// confirm m_image
		if(m_image.empty())
			return false;

		cv::blur(m_image, m_image, cv::Size(t_maskHeight | 0x00000001, t_maskHeight | 0x00000001));

		return true;
	}

	bool PZTImage::Threshold(PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) const{
		// confirm m_image
		if(m_image.empty() || m_image.channels() != 1)
			return false;

		cv::Mat thr;
		// cv::inRange Formula : dst(I)=lowerb(I) ≤ src(I) ≤ upperb(I)
		cv::inRange(m_image, cv::Scalar(t_minGray), cv::Scalar(t_maxGray), thr);
		cv::threshold(thr, thr, t_minGray - 1, 1, cv::THRESH_BINARY);

		t_reg = PZTRegions( std::move(thr) );

		return true;
	}

	// 判断 t_reg 与 m_image的尺寸是否相同。
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

	bool PZTImage::ChangeColorSpace(TransColorSpace t_color){
		// confirm m_image
		if(m_image.empty())
			return false;

		int CoverColorSpace = 0;
		bool isOK = false;
		switch (t_color){
			case TransColorSpace::TRANSCOLORSPACE_RGB2GRAY:
				CoverColorSpace = cv::COLOR_RGB2GRAY; 				isOK = PZTImage::Channels() == 3; break;

			case TransColorSpace::TRANSCOLORSPACE_BayerRGGB2RGB:	
				CoverColorSpace = cv::COLOR_BayerRGGB2RGB; 			isOK = PZTImage::Channels() == 1; break;

			case TransColorSpace::TRANSCOLORSPACE_BayerBG2RGB:
				CoverColorSpace = cv::COLOR_BayerBG2RGB;			isOK = PZTImage::Channels() == 1; break;

			case TransColorSpace::TRANSCOLORSPACE_BayerGR2RGB:
				CoverColorSpace = cv::COLOR_BayerGR2RGB;			isOK = PZTImage::Channels() == 1; break;

			case TransColorSpace::TRANSCOLORSPACE_BayerRG2RGB:
				CoverColorSpace = cv::COLOR_BayerRG2RGB;			isOK = PZTImage::Channels() == 1; break;

			case TransColorSpace::TRANSCOLORSPACE_BayerGB2RGB:
				CoverColorSpace = cv::COLOR_BayerGB2RGB;			isOK = PZTImage::Channels() == 1; break;
			
		default:
			break;
		}

		if(!isOK)
			return false;
		cv::cvtColor(m_image, m_image, CoverColorSpace);

		return true;
	}

	void PZTImage::DisplayImage(float t_factor){
		// 
		if(m_image.empty())
			return;

		cv::Mat tmp;
		cv::resize(m_image, tmp, cv::Size(m_image.cols * t_factor, m_image.rows * t_factor));
		cv::imshow("m_image", tmp);
		cv::waitKey(0);	
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

	PZTRegions::PZTRegions(const cv::Mat& t_reg) :
		m_featuresPtr(nullptr),
		m_isRegionChanged(false)
	{
		// confirm t_rg
		if(t_reg.type() != CV_8UC1)
			return;

		m_regions = t_reg.clone();
		_UpdataRegionNum();
	}

	PZTRegions::PZTRegions(cv::Mat&& t_reg){
		// confirm t_rg
		if(t_reg.type() != CV_8UC1)
			return;

		m_regions = t_reg;
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

		// Eliminate duplicate indexes
		auto indexsCpy = t_indexs;
		auto endIt = std::unique(indexsCpy.begin(), indexsCpy.end());
		indexsCpy.erase(endIt, indexsCpy.end());
	
		// updata m_regionNum
		m_regionNum = indexsCpy.size();

		// updata m_regions and m_featuresPtr
		cv::Mat tmp;
		uint32_t index = 0, idx = 0;
		m_regions = cv::Mat::zeros(t_reg.m_regions.rows, t_reg.m_regions.cols, CV_8UC1);
		if(t_reg.m_featuresPtr != nullptr){

			m_featuresPtr = std::make_shared<std::vector<RegionFeature>>();
			m_featuresPtr->reserve(m_regionNum);

			for(idx = 0; idx < m_regionNum; ++idx){
				index = indexsCpy[idx];
				m_featuresPtr->push_back( (*t_reg.m_featuresPtr)[index] );
				cv::inRange(t_reg.m_regions, cv::Scalar(index + 1), cv::Scalar(index + 1), tmp);
				cv::threshold(tmp, tmp, index, 1, cv::THRESH_BINARY);
				tmp = tmp + index;		
				m_regions += tmp;
			}

		}else{

			for(idx = 0; idx < m_regionNum; ++idx){
				index = indexsCpy[idx];
				cv::inRange(t_reg.m_regions, cv::Scalar(index + 1), cv::Scalar(index + 1), tmp);
				cv::threshold(tmp, tmp, index, 1, cv::THRESH_BINARY);
				tmp = tmp + index;	
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

	// 最后connection
	PZTRegions PZTRegions::operator + (const PZTRegions& t_other) const {
		// confirm t_other and *this
		cv::Size2i selfSize, otherSize;
        this->GetRegionSize(selfSize);
		t_other.GetRegionSize(otherSize);
        if(selfSize.height != otherSize.height || selfSize.width != otherSize.width)
            return PZTRegions();

		//
		if(t_other.Empty())
			return PZTRegions(*this);

		cv::Mat regs = m_regions + t_other.m_regions;
		cv::threshold(regs, regs, 0, 1, cv::THRESH_BINARY);

		PZTRegions obj(std::move(regs));
		obj.Connection();

		return std::move(obj);
	}

	bool PZTRegions::Clear(){
		cv::threshold(m_regions, m_regions, 0, 0, cv::THRESH_BINARY);
		m_regionNum = 0;
		m_isRegionChanged = true;

		return true;
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

	bool PZTRegions::GetRegionSize(cv::Size2i& t_size) const{
		// confirm m_regions
		if(m_regions.empty()){
			t_size = cv::Size2i(0, 0);
			return false;
		}
		
		t_size = cv::Size2i(m_regions.cols, m_regions.rows);

		return true;
	}

	float PZTRegions::GetRegionFeature(unsigned int t_index, FeatureType t_type){
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
	
	// warpAffine()
	bool PZTRegions::MoveRegion(int t_row, int t_col){
		// confirm m_regions
		if(m_regions.empty())
			return false;

		cv::Matx23f m(1, 0, t_col, 0, 1, t_row);
		cv::warpAffine(m_regions, m_regions, m, cv::Size(m_regions.cols, m_regions.rows), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

		return true;
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

		//cv::InputArray
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

	bool PZTRegions::ShapeTrans(ShapeTransType t_type){
		// 多边形拟合(简化后续计算量) -> 

		std::vector<std::vector<cv::Point>> contours;
    	cv::findContours(m_regions, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

		PZTRegions::Clear();

		int regNum = contours.size();
		std::vector<std::vector<cv::Point>> contours_poly(regNum);
		std::vector<cv::Rect> boundRect( regNum );
		
		for(int i = 0; i < regNum; ++i){
			cv::approxPolyDP(contours[i], contours_poly[i], APPROXIMATION_ACCURACY, true);

			switch(t_type){
				case ShapeTransType::SHAPETRANSTYPE_RECTANGLE1:
					boundRect[i] = cv::boundingRect( contours_poly[i] ); 
					cv::rectangle(m_regions, boundRect[i], cv::Scalar(1), 1); break;

				default: break;
			}
		}

		//
		bool res = false;
		m_regionNum = 1;
		res = PZTRegions::FillUp();
		res = PZTRegions::Connection();

		return res;
	}

	bool PZTRegions::Erosion(StructElement t_elm, unsigned int t_kernelWidth, unsigned int t_kernelHeight){
		if(m_regionNum == 0)
			return true;

		// confirm m_regions and m_regionNum
		if(m_regions.empty() || m_regionNum != 1)
			return false;

		unsigned int kernelWidth= t_kernelWidth | 0x00000001;
		unsigned int kernelHeight= t_kernelHeight | 0x00000001;
		cv::Mat kernel = cv::getStructuringElement(static_cast<cv::MorphShapes>(t_elm), cv::Size(kernelWidth, kernelHeight));
		cv::erode(m_regions, m_regions, kernel);

		m_isRegionChanged = true;

		return true;
	}

	bool PZTRegions::Dilation(StructElement t_elm, unsigned int t_kernelWidth, unsigned int t_kernelHeight){
		if(m_regionNum == 0)
			return true;

		// confirm m_regions and m_regionNum
		if(m_regions.empty() || m_regionNum != 1)
			return false;

		unsigned int kernelWidth= t_kernelWidth | 0x00000001;
		unsigned int kernelHeight= t_kernelHeight | 0x00000001;
		cv::Mat kernel = cv::getStructuringElement(static_cast<cv::MorphShapes>(t_elm), cv::Size(kernelWidth, kernelHeight));
		cv::dilate(m_regions, m_regions, kernel);

		m_isRegionChanged = true;

		return true;
	}

	bool PZTRegions::Opening(StructElement t_elm, unsigned int t_kernelWidth, unsigned int t_kernelHeight){
		if(m_regionNum == 0)
			return true;

		// confirm m_regions and m_regionNum
		if(m_regions.empty() || m_regionNum != 1)
			return false;

		unsigned int kernelWidth= t_kernelWidth | 0x00000001;
		unsigned int kernelHeight= t_kernelHeight | 0x00000001;
		cv::Mat kernel = cv::getStructuringElement(static_cast<cv::MorphShapes>(t_elm), cv::Size(kernelWidth, kernelHeight));
		cv::morphologyEx(m_regions, m_regions, cv::MORPH_OPEN, kernel);

		m_isRegionChanged = true;

		return true;
	}

	unsigned int PZTRegions::GetRegionNum() const{
		return m_regionNum;
	}

	void PZTRegions::DisplayRegion(float t_factor){
		if (m_regions.empty())
			return;

		cv::Mat tmp;
		cv::resize(m_regions, tmp, cv::Size(m_regions.cols * t_factor, m_regions.rows * t_factor));
		cv::imshow("m_regions", tmp * 40);
		cv::waitKey(0);
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
	
	class Base {
	public:
		enum {
			row = 10,
			col = 11
		} ;

		Base() {
			int b = 1 + row;
		}

		static int roww;
		static int colw;
	};

	int Base::roww = 11;
	int Base::colw = 11;

	bool TestCore() {
		bool res = false;

		// 测试 trans_shape() 
/*
		cv::Mat reg = cv::imread("./connectedDomain.jpg");
		cv::cvtColor(reg, reg, cv::COLOR_RGB2GRAY);

		cv::threshold(reg, reg, 100, 1, cv::THRESH_BINARY);

		PZTRegions obj(reg);
		obj.DisplayRegion();

		obj.ShapeTrans(ShapeTransType::SHAPETRANSTYPE_RECTANGLE1);
		obj.DisplayRegion();

		obj.MoveRegion(50, 60);
		obj.DisplayRegion();

/*
		obj.Connection();
		//obj.DisplayRegion();

		//RegionFeature feature = obj.GetRegionFeature(0);

		// 测试 threshold()
		PZTImage img("./connectedDomain.jpg");

		PZTImage imgO;
		PZTRegions regO;
*/
		// 测试 cvtColor() 输入bayer图像是否为 3 通道
		//auto imgI = cv::imread("./connectedDomain.jpg");

		//cv::Mat imgO(imgI);

		//cv::threshold(imgO, imgO, 122, 255, cv::THRESH_BINARY_INV);

		//cv::imshow("imgO", imgI);
		//cv::waitKey(0);

		Base a;
		std::cout << sizeof(a);

		cv::Mat reg = cv::imread("./connectedDomain.jpg");
		cv::_InputArray bb(reg);

		bb.size();

		return res;
	}

}
