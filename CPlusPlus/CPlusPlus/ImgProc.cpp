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
		
		t_imgI.ChangeColorSpace(TRANSCOLORSPACE_RGB2GRAY);
		t_imgO = t_imgI;

		return res;
	}
	
	bool OperatorSet::threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;
		
		cv::Mat i_image = t_imgI.m_image;
		cv::Mat o_image ,o_image1 ,o_image2;

		cv::threshold(i_image, o_image1, t_minGray, 1, cv::THRESH_BINARY);
		cv::threshold(i_image, o_image2, t_maxGray, 1, cv::THRESH_BINARY_INV);			//阈值分割，分割后连通域只有一种,像素值一致为1
		
		bitwise_and(o_image1, o_image2, o_image);
		PZTRegions o_region(o_image);												//利用图像构造region，得到阈值分割结果图像
		t_reg = o_region;															//输出

		
		return res;
	}

	bool OperatorSet::connection(PZTRegions t_reg, PZTRegions& t_regs) {
		bool res = true;

		t_reg.Connection();				//连通域分割，已得到m_regionNum,m_regions，缺少m_feature
		for (int i = 1; i < t_reg.GetRegionNum(); i++)
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
		if(indexs.empty())
		{
			std::cout << "数量为0" << std::endl;
			return false;
		}
		//t_regI.DisplayRegion();
		PZTRegions result(t_regI, indexs);
		t_regO = result;

		return res;
	}

	bool OperatorSet::emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor) {
		bool res = true;

		if (t_imgI.m_image.empty())
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

	bool OperatorSet::get_image_size(PZTImage t_imgI, unsigned int& width, unsigned int& height)
	{
		bool res = true;
		if (t_imgI.m_image.empty())
			return false;
		t_imgI.GetImageSize(width,height);

		return res;
	}

	bool OperatorSet::decompose3(PZTImage t_imgI, PZTImage& t_imgR, PZTImage& t_imgG, PZTImage& t_imgB)
	{
		bool res = true;
		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() != CV_8UC3)
			return false;
		t_imgI.Decompose(t_imgR, t_imgG, t_imgB);

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

	bool OperatorSet::dyn_threshold(PZTImage t_imgI, PZTImage t_thresholdimgI, PZTRegions& t_regO, uint8_t t_offset, Light_Dark Light_Dark)
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
		//cv::Mat result = t_imgO.m_image;			此处报异常
		cv::Mat result = cv::Mat(src.size(), CV_8UC1, cv::Scalar(0));

		int r = src.rows; //高
		int c = src.cols; //宽
		int Value = 0;
		for (int i = 0; i < c*r; i++)
		{
			int datasrc = src.data[i];
			int datasrcMean = srcMean.data[i];
			
			switch (Light_Dark)
			{
			case Light:
				Value = datasrc - datasrcMean;
				if (Value >= t_offset)
				{
					result.data[i] = 1;
				}
				break;
			case Dark:
				Value = datasrcMean - datasrc;
				if (Value >= t_offset)
				{
					result.data[i] = 1;
				}
				break;
			case Equal:
				Value = datasrc - datasrcMean;
				if (Value >= -t_offset && Value <= t_offset)
				{
					result.data[i] = 1;
				}
				break;
			case Not_equal:
				Value = datasrc - datasrcMean;
				if (Value < -t_offset || Value > t_offset)
				{
					result.data[i] = 1;
				}
				break;
			default:
				break;
			}
		}
		PZTRegions res_region(result);
		t_regO = res_region;

		return res;
	}

	bool OperatorSet::opening_rectangle1(PZTRegions t_regI, PZTRegions& t_regO, uint8_t t_Width, uint8_t t_Height)
	{
		bool res = true;
		t_regI.Opening(STRUCTELEMENT_RECTANGLE, t_Width, t_Height);
		t_regO = t_regI;
		
		return res;
	}

	bool OperatorSet::opening_circle(PZTRegions t_regI, PZTRegions& t_regO, uint8_t radius)
	{
		bool res = true;
		t_regI.Opening(STRUCTELEMENT_CIRCLE, radius, radius);
		t_regO = t_regI;

		return res;
	}
	bool OperatorSet::dilation_rectangle1(PZTRegions t_regI, PZTRegions& t_regO, uint8_t t_Width, uint8_t t_Height)
	{
		bool res = true;
		t_regI.Dilation(STRUCTELEMENT_RECTANGLE, t_Width, t_Height);
		t_regO = t_regI;

		return res;
	}

	bool OperatorSet::intersection(PZTRegions t_regI1, PZTRegions t_regI2, PZTRegions& t_regO)
	{
		bool res = true;


		return res;
	}

	bool OperatorSet::move_region(PZTRegions t_regI, PZTRegions& t_regO, int rows, int cols)
	{
		bool res = true;
		t_regI.MoveRegion(rows, cols);
		t_regO = t_regI;

		return res;
	}

	bool OperatorSet::union1(PZTRegions t_regI, PZTRegions& t_regO)
	{
		bool res = true;
		t_regI.Disconnection();
		t_regO = t_regI;

		return res;
	}

	bool OperatorSet::union2(PZTRegions t_regI1, PZTRegions t_regI2, PZTRegions& t_regO)
	{
		bool res = true;

		return res;
	}

	bool OperatorSet::shape_trans(PZTRegions t_regI, PZTRegions& t_regO, ShapeTransType t_type)
	{
		bool res = true;
		t_regI.ShapeTrans(t_type);
		t_regO = t_regI;

		return res;
	}

	bool OperatorSet::region_features(PZTRegions t_regI, Features t_fea, int &Value)
	{
		bool res = true;

		int num = t_regI.GetRegionNum();
		for (int i = 1; i < num; i++)
		{
			RegionFeature regf = t_regI.GetRegionFeature(i);
			switch (t_fea)
			{
			case PZTIMAGE::FEATURES_AREA:
				//面积计算
				Value = regf.m_area;
				break;
			case PZTIMAGE::FEATURES_CIRCULARITY:
				//矩形度计算
				Value = regf.m_circularity;
				break;
			case PZTIMAGE::FEATURES_ROW:
				//行计算
				Value = regf.m_row;
				break;
			case PZTIMAGE::FEATURES_COLUMN:
				//列计算
				Value = regf.m_col;
				break;
			default:
				break;
			}
		}

		return res;
	}

	bool OperatorSet::difference(PZTRegions t_regI1, PZTRegions t_regI2, PZTRegions& t_regO)
	{
		bool res = true;


		return res;
	}

	bool OperatorSet::display_image(PZTImage t_imgI,std::string WindowName,bool save,bool diplay)
	{
		bool res = true;
		if (t_imgI.m_image.empty())
			return false;
		cv::Mat show_image;
		if (diplay)
			show_image = t_imgI.m_image;
		else
			show_image = t_imgI.m_mask;
		if (save)
			cv::imwrite("./save.jpg", show_image);
		int width = show_image.size().width;
		int height = show_image.size().height;
		if (width > 1920 || height > 1080)
		{
			int w = width / 1920 + 2;
			int h = height / 1080 + 2;
			if (w > h)
			{
				cv::resize(show_image, show_image, cv::Size(int(width / w), int(height / w)));
			}
			else
			{
				cv::resize(show_image, show_image, cv::Size(int(width / h), int(height / h)));
			}
			cv::imshow(WindowName, show_image);
		}
		else
		{
			cv::imshow(WindowName, show_image);
		}
	
		int keyValue = cv::waitKey(10);

		if (keyValue && 0xFF == '27')
			exit(0);
		else
			cv::waitKey(0);

		return res;
	}

	bool OperatorSet::display_image(PZTImage t_imgI) {
		bool res = true;

		cv::imshow("test", 60* t_imgI.m_mask);
		cv::waitKey(0);
		return res;
	}

	bool OperatorSet::display_region(PZTRegions t_regI, float mutiple)
	{
		bool res = true;

		t_regI.DisplayRegion(mutiple);

		return res;
	}

	bool TestImgProc() {
		bool res = false;

		////PZTImage img("E:\\1dong\\5-18-ExposureTime3000-normal\\IMG_Light\\BX2.tif");
		//std::string filename = "E:/img/1122/1122/25.tif";
		//
		//PZTImage MiniLED,Gray,Mean,Reduce;
		//PZTRegions Threshold_Region,Fill,Open,Connection,Trans,Dilation,Union,Move,Difference;
		//OperatorSet::read_image(MiniLED, filename);
		//OperatorSet::gray_image(MiniLED, Gray);
		//OperatorSet::threshold(Gray, Threshold_Region, 180, 255);
		////OperatorSet::fill_up(Threshold_Region, Fill);												//不报错没显示
		//OperatorSet::opening_rectangle1(Threshold_Region, Open, 30, 30);
		//OperatorSet::connection(Open, Connection);
		//OperatorSet::shape_trans(Connection, Trans, ShapeTransType::SHAPETRANSTYPE_RECTANGLE1);		
		////OperatorSet::dilation_rectangle1(Open, Dilation, 120, 120);								//若先进行connection,无效果
		//
		////OperatorSet::union1(Connection, Union);
		////OperatorSet::move_region(Union, Move, 1700, 1700);
		////OperatorSet::reduce_domain(Gray, Union, Reduce);											//无效果

		////OperatorSet::mean_image(Gray, Mean, 5, 5);

		////OperatorSet::display_image(Reduce, "Reduce" ,true);
		//OperatorSet::display_region(Trans, 0.1);
		////OperatorSet::display_region(Dilation, 0.1);

		///***************************复现*******************************/
		//std::string filename = "E:/img/1122/1122/26.tif";
		//
		//PZTImage Image ,Gray ,Mean;
		//PZTRegions Background ,MiniLED , Connection,Union, Dynthreshold, Opening,Move;
		//
		////阈值分割数据
		//uint8_t background = 120;		//背景的最大阈值
		//uint8_t miniled = 170;			//芯片反光区域的最小阈值

		////预处理
		//OperatorSet::read_image(Image, filename);
		//OperatorSet::gray_image(Image, Gray);
		////OperatorSet::threshold(Gray, Background, 0, 120);
		//OperatorSet::threshold(Gray, MiniLED, 170, 255);

		////边界确定   -->   芯片剔除
		////OperatorSet::connection(MiniLED, Connection);		//connection之后Num还是1,原因是Num数量超出255，uchar的限制
		////OperatorSet::select_shape(Background, Background, FEATURES_AREA, 120000, 99999999);	//出错,原因为indexs为0无法构造
		////OperatorSet::union1(Connection, Union);
		//OperatorSet::mean_image(Gray, Mean, 30, 30);
		//OperatorSet::dyn_threshold(Gray, Mean, Dynthreshold, 7, Dark);
		//OperatorSet::opening_rectangle1(Dynthreshold, Opening,10,10);		//无效果,原因为dyn时设置的灰度值为255，改为1
		////OperatorSet::move_region(Dynthreshold, Move, 1000, 1000);

		////效果显示
		//OperatorSet::display_image(Mean, "mean", 0,1);
		////OperatorSet::display_region(Connection, 0.1);
		//OperatorSet::display_region(Opening, 0.1);

		/***************复现简单图像(缺陷数量少的图像)******************/
		//变量部分
		std::string filename = "E:/1dong/5-18-ExposureTime3000-normal/test.png";
		unsigned int width, height;

		//图像部分
		PZTImage Image,Gray,R,G,B,Emphasize,Reduce,GrayReduce;
		PZTRegions Region,Threshold,Connection,SelectRegion,Union,Test;

		//算子部分
		OperatorSet::read_image(Image, filename);
		//OperatorSet::gray_image(Image, Gray);
		OperatorSet::get_image_size(Image, width, height);
		OperatorSet::decompose3(Image, R, G, B);
		OperatorSet::emphasize(B, Emphasize, 36, 36, 3);
		OperatorSet::threshold(Emphasize, Threshold, 200, 255);
		//先做形态学再打散可能少一些缺陷，可能不会出现超出情况
		OperatorSet::connection(Threshold, Connection);
		OperatorSet::select_shape(Connection, SelectRegion, FEATURES_AREA, 100, 1000);
		OperatorSet::union1(Connection, Union);
		OperatorSet::reduce_domain(B, Union, Reduce);

		//显示部分
		//OperatorSet::display_image(Reduce);
		//OperatorSet::display_image(Reduce, "image", 0, 0);
		OperatorSet::display_region(SelectRegion, 1);
		OperatorSet::display_region(Connection, 1);

		return res;
	}

}