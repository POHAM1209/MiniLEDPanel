#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:读取图片
	* param0[o]:读取到的图片		任意通道数图像
	* param1[i]:图片路径			string
	*/
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

	/*brief:阈值分割
	* param0[i]:输入image			灰度图像
	* param1[o]:输出region			二值化图像，0或者255
	* param2[i]:最小阈值
	* param3[i]:最大阈值

	*/
	bool OperatorSet::threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		cv::Mat i_image = t_imgI.m_mask;
		cv::Mat o_image;

		cv::threshold(i_image, o_image, t_minGray, t_maxGray, cv::THRESH_BINARY);	//阈值分割，分割后连通域只有一种
		PZTRegions o_region(o_image);												//利用图像构造region，得到阈值分割结果图像
		t_reg = o_region;															//输出

		
		return res;
	}

	/*brief:连通域分割
	* param0[i]:输入region
	* param1[o]:输出region(连通域分开)
	*/
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

	/*brief:在image上减去region
	* param0[i]:输入image
	* param1[i]:输入region
	* param2[o]:输出image
	*/
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

	/*brief:按照形状特征选择结果
	* param0[i]:输入region，一般为connection后的region
	* param1[o]:输出region
	* param2[i]:输入特征类型(以什么特征选择)
	* param3[i]:最小范围
	* param4[i]:最大范围
	*/
	bool OperatorSet::select_shape(PZTRegions t_regI, PZTRegions& t_regO, Features t_fea, float t_min, float t_max) {
		bool res = true;


		//不能用region来判断空
		//if (t_regI.m_regions.empty())
		//	return false;

		int num = t_regI.GetRegionNum();
		if (t_fea == FEATURES_AREA)
		{
			//做法：遍历region连通域，并计算面积
			//问题：面积不在要求内的连通域灰度值赋0
			for (int i = 1; i < num; i++)//0是背景？是从0还是从1开始遍历
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_area >= t_min && regf.m_area <= t_max)
					continue;
				else
				{
					//如何将不符合的灰度值置0? 怎么拿到region位置来置0?

				}
			}
		}
		else if (t_fea == FEATURES_CIRCULARITY)
		{
			for (int i = 1; i < num; i++)//0是背景？是从0还是从1开始遍历
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_circularity >= t_min && regf.m_circularity <= t_max)
					continue;
				else
				{
					//如何将不符合的灰度值置0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else if (t_fea == FEATURES_ROW)
		{
			for (int i = 1; i < num; i++)//0是背景？是从0还是从1开始遍历
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_row >= t_min && regf.m_row <= t_max)
					continue;
				else
				{
					//如何将不符合的灰度值置0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else if (t_fea == FEATURES_COLUMN)
		{
			for (int i = 1; i < num; i++)//0是背景？是从0还是从1开始遍历
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_col >= t_min && regf.m_col <= t_max)
					continue;
				else
				{

					//如何将不符合的灰度值置0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else
		{
			std::cout << "error! there is not this Feature!" << std::endl;
		}

		return res;
	}

	/*brief:增强图片对比度
	* param0[i]:输入图片
	* param1[o]:输出图片
	* param2[i]:mask长
	* param3[i]:mask宽
	* param4[i]:增强强度
	*/
	bool OperatorSet::emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		//公式res := round((orig - mean) * Factor) + orig
		//等价于在MaskHeight、MaskWidth的空间内中心化后增加方差
		cv::Mat mean;

		//以单通道为准，m_mask
		for (int i = 0; i < t_imgI.m_image.rows; i++)
		{
			const uchar* rptr = t_imgI.m_image.ptr<uchar>(i);
			uchar* mptr = mean.ptr<uchar>(i);
			uchar* optr = t_imgO.m_image.ptr<uchar>(i);
			for (int j = 0; j < t_imgI.m_image.cols; j++)
			{
				optr[j] = cv::saturate_cast<uchar>(round((rptr[j] - mptr[j]) * Factor) + float(rptr[j]) * 1.0f);
			}
		}

		return res;
	}

	/*brief:灰度增强
	* param0[i]:输入图片
	* param1[o]:输出图片
	* param2[i]:mask长
	* param3[i]:mask宽
	*/
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

	bool TestImgProc() {
		bool res = false;

		PZTImage img("E:\\1dong\\5-18-ExposureTime3000-normal\\IMG_Light\\BX2.tif");
		


		return res;
	}

}