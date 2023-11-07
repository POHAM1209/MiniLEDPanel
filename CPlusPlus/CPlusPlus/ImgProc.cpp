#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:读取图片
	* param0[o]:读取到的图片		任意通道数图像
	* param1[i]:图片路径			string
	*/
	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = false;
		PZTImage readimage(t_fileName);					//利用文件名构造函数，读取图像
		t_imgO = readimage;								//输出
		return res;
	}

	/*brief:阈值分割
	* param0[i]:输入image			灰度图像
	* param1[o]:输出region			二值化图像，0或者255
	* param2[i]:最小阈值
	* param3[i]:最大阈值
	*/
	bool OperatorSet::threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = false;
		cv::Mat i_image = t_img.m_image;
		cv::Mat o_image;
		cv::threshold(i_image, o_image, t_minGray, t_maxGray, cv::THRESH_BINARY);	//阈值分割，分割后连通域只有一种
		PZTRegions n_region(o_image);												//利用图像构造region，得到阈值分割结果图像
		t_reg = n_region;															//输出
		return res;
	}

	/*brief:连通域分割
	* param0[i]:输入region
	* param1[o]:输出region(连通域分开)
	*/
	bool OperatorSet::connection(PZTRegions t_reg, PZTRegions& t_regs) {
		bool res = false;
		t_reg.Connection();				//连通域分割
		t_reg.GetRegionFeature(0);		//获取最新的region feature，需遍历
		t_regs = t_reg;					//输出
		return res;
	}

	/*brief:在image上减去region
	* param0[i]:输入image
	* param1[i]:输入region
	* param2[o]:输出image
	*/
	bool OperatorSet::reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO) {
		bool res = false;
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
		bool res = false;

		if (t_fea == FEATURES_AREA)
		{
			//做法：遍历region连通域，并计算面积
			//面积不在要求内的连通域灰度值赋0
			//问题：怎么计算面积？
			for (i = 0; i < t_regI.m_FeatureNum; i++)
			{
				RegionFeature i = t_regI.GetRegionFeature(i);
			}
		}
		else if (t_fea == FEATURES_CIRCULARITY)
		{

		}
		else if (t_fea == FEATURES_ROW)
		{

		}
		else if (t_fea == FEATURES_COLUMN)
		{

		}
		else
		{
			std::cout << "error! there is not this feature!" << std::endl;
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
		bool res = false;
		//公式res := round((orig - mean) * Factor) + orig
		//等价于在MaskHeight、MaskWidth的空间内中心化后增加方差
		cv::Mat mean;

		//等价于求指定范围窗口内的均值
		cv::blur(t_imgI.m_image, mean, cv::Size(t_MaskWidth, t_MaskHeight));
		t_imgO.m_image.create(t_imgI.m_image.size(), t_imgI.m_image.type());
		if (t_imgI.m_image.type() == CV_8UC1)
		{
			for (int i = 0; i < t_imgI.m_image.rows; i++)
			{
				const uchar* rptr = t_imgI.m_image.ptr<uchar>(i);
				uchar* mptr = mean.ptr<uchar>(i);
				uchar* optr = t_imgO.m_image.ptr<uchar>(i);
				for (int j = 0; j < t_imgI.m_image.cols; j++)
				{
					optr[j] = cv::saturate_cast<uchar>(round((rptr[j] - mptr[j]) * Factor) + rptr[j] * 1.0f);
				}
			}
		}
		else if (t_imgI.m_image.type() == CV_8UC3)
		{
			for (int i = 0; i < t_imgI.m_image.rows; i++)
			{
				const uchar* rptr = t_imgI.m_image.ptr<uchar>(i);
				uchar* mptr = mean.ptr<uchar>(i);
				uchar* optr = t_imgO.m_image.ptr<uchar>(i);
				for (int j = 0; j < t_imgI.m_image.cols; j++)
				{
					//饱和转换 小于0的值会被置为0 大于255的值会被置为255
					optr[j * 3] = cv::saturate_cast<uchar>(round((rptr[j * 3] - mptr[j * 3]) * Factor) + rptr[j * 3] * 1.0f);
					optr[j * 3 + 1] = cv::saturate_cast<uchar>(round((rptr[j * 3 + 1] - mptr[j * 3 + 1]) * Factor) + rptr[j * 3 + 1] * 1.0f);
					optr[j * 3 + 2] = cv::saturate_cast<uchar>(round((rptr[j * 3 + 2] - mptr[j * 3 + 2]) * Factor) + rptr[j * 3 + 2] * 1.0f);
				}
			}
		}

		return res;
	}



	bool TestImgProc() {
		bool res = false;

		return res;
	}

}