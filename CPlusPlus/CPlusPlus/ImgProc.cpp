#include "ImgProc.h"

namespace PZTIMAGE {

	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = true;
		if (t_fileName.empty())
			return false;

		PZTImage readimage(t_fileName);					//利用文件名构造函数，读取图像
		t_imgO = readimage;								//输出

		return res;
	}

	bool OperatorSet::gray_image(PZTImage t_imgI, PZTImage& t_imgO)
	{
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC1)
			return false;
		
		t_imgI.RGB2Gray();
		t_imgO = t_imgI;

		return res;
	}

	bool OperatorSet::threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;

		cv::Mat i_image = t_imgI.m_image;
		cv::Mat o_image;

		cv::threshold(i_image, o_image, t_minGray, t_maxGray, cv::THRESH_BINARY);	//阈值分割，分割后连通域只有一种
		PZTRegions o_region(o_image);												//利用图像构造region，得到阈值分割结果图像
		t_reg = o_region;															//输出

		
		return res;
	}

	bool OperatorSet::connection(PZTRegions t_reg, PZTRegions& t_regs) {
		bool res = true;

		t_reg.Connection();				//连通域分割，已得到m_regionNum,m_regions，缺少m_feature
		for (int i = 0; i < t_reg.GetRegionNum(); i++)
		{
			//需要变量来接收，还是直接在函数里改了? 
			t_reg.GetRegionFeature(i);
		}
		t_regs = t_reg;					//输出
		return res;
	}

	bool OperatorSet::reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;


		//做法：返回image，所以直接container相减？等于是矩阵相减
		t_imgI.ReduceDomain(t_reg);
		t_imgO = t_imgI;
		return res;
	}

	bool OperatorSet::select_shape(PZTRegions t_regI, PZTRegions& t_regO, Features t_fea, float t_min, float t_max) {
		bool res = true;

		std::vector<uint32_t> indexs;
		int num = t_regI.GetRegionNum();

		for (int i = 1; i < num; i++)
		{
			RegionFeature regf = t_regI.GetRegionFeature(i);
			switch (t_fea)
			{
			case PZTIMAGE::FEATURES_AREA:
				if (regf.m_area >= t_min && regf.m_area <= t_max)
				{
					indexs.push_back(i);
				}
				break;
			case PZTIMAGE::FEATURES_CIRCULARITY:
				if (regf.m_circularity >= t_min && regf.m_circularity <= t_max)
				{
					indexs.push_back(i);
				}
				break;
			case PZTIMAGE::FEATURES_ROW:
				if (regf.m_row >= t_min && regf.m_row <= t_max)
				{
					indexs.push_back(i);
				}
				break;
			case PZTIMAGE::FEATURES_COLUMN:
				if (regf.m_col >= t_min && regf.m_col <= t_max)
				{
					indexs.push_back(i);
				}
				break;
			default:
				break;
			}
		}
		PZTRegions result(t_regI, indexs);
		t_regO = result;

		return res;
	}

	bool OperatorSet::emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		//公式res := round((orig - mean) * Factor) + orig
		//等价于在MaskHeight、MaskWidth的空间内中心化后增加方差
		cv::Mat mean;
		cv::Mat input = t_imgI.m_image;
		cv::Mat output;

		cv::blur(input, mean, cv::Size(t_MaskWidth, t_MaskHeight));
		output.create(input.size(), input.type());

		if (input.type() == CV_8UC1)
		{
		    for (int i = 0; i < input.rows; i++)
		    {
		        const uchar* rptr = input.ptr<uchar>(i);
		        uchar* mptr = mean.ptr<uchar>(i);
		        uchar* optr = output.ptr<uchar>(i);
		        for (int j = 0; j < input.cols; j++)
		        {
		            optr[j] = cv::saturate_cast<uchar>(round((rptr[j] - mptr[j]) * Factor) + rptr[j] * 1.0f);
		        }
		    }
		}
		else if (input.type() == CV_8UC3)
		{
		    for (int i = 0; i < input.rows; i++)
		    {
		        const uchar* rptr = input.ptr<uchar>(i);
		        uchar* mptr = mean.ptr<uchar>(i);
		        uchar* optr = output.ptr<uchar>(i);
		        for (int j = 0; j < input.cols; j++)
		        {
		            //饱和转换 小于0的值会被置为0 大于255的值会被置为255
		            optr[j * 3] = cv::saturate_cast<uchar>(round((rptr[j * 3] - mptr[j * 3]) * Factor) + rptr[j * 3] * 1.0f);
		            optr[j * 3 + 1] = cv::saturate_cast<uchar>(round((rptr[j * 3 + 1] - mptr[j * 3 + 1]) * Factor) + rptr[j * 3 + 1] * 1.0f);
		            optr[j * 3 + 2] = cv::saturate_cast<uchar>(round((rptr[j * 3 + 2] - mptr[j * 3 + 2]) * Factor) + rptr[j * 3 + 2] * 1.0f);
		        }
		    }
		}
		t_imgO.m_image = output;

		return res;
	}

	bool OperatorSet::gray_range_rect(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight)
	{
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		//图像边界扩充
		int hh = (t_MaskHeight - 1) / 2;
		int hw = (t_MaskWidth - 1) / 2;
		cv::Mat Newsrc;
		cv::copyMakeBorder(t_imgI.m_mask, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
		t_imgO.m_mask= cv::Mat::zeros(t_imgI.m_mask.rows, t_imgI.m_mask.cols, t_imgI.m_mask.type());

		//遍历图像
		for (int i = 0; i < t_imgI.m_mask.rows; i++)
		{
			for (int j = 0; j < t_imgI.m_mask.cols; j++)
			{
				//uchar srcValue = src.at<uchar>(i, j);
				int minValue = 255;
				int maxValue = 0;
				for (int k = 0; k < hh; k++)
				{
					for (int z = 0; z < hw; z++)
					{
						int srcValue = (int)Newsrc.at<uchar>(i + k, j + z);
						minValue = minValue > srcValue ? srcValue : minValue;
						maxValue = maxValue > srcValue ? maxValue : srcValue;

					}
				}
				uchar diffValue = (uchar)(maxValue - minValue);
				t_imgO.m_mask.at<uchar>(i, j) = diffValue;
			}
		}

		return res;
	}

	bool OperatorSet::area_center(PZTRegions t_regI, int& t_area, cv::Point2f& t_point)
	{
		bool res = true;

		return res;
	}

	bool OperatorSet::fill_up(PZTRegions t_regI, PZTRegions t_regO)
	{
		bool res = true;

		//有没有判断点
		t_regI.FillUp();
		t_regO = t_regI;

		return res;
	}

	bool OperatorSet::erosion_circle(PZTRegions t_regI, PZTRegions& t_regO, int t_radius)
	{
		bool res = true;

		StructElement type = STRUCTELEMENT_CIRCLE;
		t_regI.Erosion(type,t_radius);
		t_regO = t_regI;

		return res;
	}
	
	bool OperatorSet::mean_image(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight)
	{
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		cv::Mat input = t_imgI.m_image;
		cv::Mat blur_image;
		cv::blur(input, blur_image, cv::Size(t_MaskWidth, t_MaskHeight));
		t_imgO.m_image = blur_image;

		return res;
	}

	bool OperatorSet::dyn_threshold(PZTImage t_imgI, PZTImage t_thresholdimgI, PZTImage& t_imgO, uint8_t t_offset, LightDark Light_Dark)
	{
		bool res = true;
		//使用Opencv实现Halcon中的动态阈值
		//src是原图,灰度图
		//srcMean是平滑滤波之后的图
		//最好不要把Offset这个变量设置为0，因为这样会导致最后找到太多很小的regions，而这基本上都是噪声。
		//所以这个值最好是在5-40之间，值选择的越大，提取出来的regions就会越小。

		//判断
		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;
		if (t_thresholdimgI.m_image.empty())
			return false;
		if (t_thresholdimgI.m_image.type() == CV_8UC3)
			return false;

		cv::Mat src = t_imgI.m_image;
		cv::Mat srcMean = t_thresholdimgI.m_image;
		cv::Mat result = t_imgO.m_image;

		int r = src.rows; //高
		int c = src.cols; //宽
		int Value = 0;
		for (int i = 0; i < r; i++)
		{
			uchar* datasrc = src.ptr<uchar>(i); //指针访问图像像素
			uchar* datasrcMean = srcMean.ptr<uchar>(i);
			uchar* dataresult = result.ptr<uchar>(i);

			for (int j = 0; j < c; j++)
			{
				switch (Light_Dark)
				{
				case Light:
					Value = datasrc[j] - datasrcMean[j];
					if (Value >= t_offset)
					{
						dataresult[j] = 255;
					}
					break;
				case Dark:
					Value = datasrcMean[j] - datasrc[j];
					if (Value >= t_offset)
					{
						dataresult[j] = 255;
					}
					break;
				case Equal:
					Value = datasrc[j] - datasrcMean[j];
					if (Value >= -t_offset && Value <= t_offset)
					{
						dataresult[j] = 255;
					}
					break;
				case Not_equal:
					Value = datasrc[j] - datasrcMean[j];
					if (Value < -t_offset || Value > t_offset)
					{
						dataresult[j] = 255;
					}
					break;
				default:
					break;
				}
			}
		}
		t_imgO.m_image = result;

		return res;
	}

	bool OperatorSet::dev_display(PZTImage t_imgI,std::string WindowName)
	{
		bool res = true;
		if (t_imgI.m_image.empty())
			return false;

		cv::imshow(WindowName, t_imgI.m_image);

		
		int keyValue = cv::waitKey(10);

		if (keyValue && 0xFF == '27')
			exit(0);
		else
			cv::waitKey(0);

		return res;
	}

	bool TestImgProc() {
		bool res = false;

		//PZTImage img("E:\\1dong\\5-18-ExposureTime3000-normal\\IMG_Light\\BX2.tif");
		std::string filename = "E:/img/1122/1122/30.tif";
		
		PZTImage MiniLED,Gray;
		PZTRegions Threshold_Region;
		OperatorSet::read_image(MiniLED, filename);
		OperatorSet::gray_image(MiniLED, Gray);
		OperatorSet::threshold(Gray, Threshold_Region, 144, 255);
		OperatorSet::dev_display(MiniLED, "MiniLED");

		return res;
	}

}