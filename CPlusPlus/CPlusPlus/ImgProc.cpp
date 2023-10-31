#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:读取图片
	* param0[o]:读取到的图片
	* param1[i]:图片路径
	*/
	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = false;
		PZTImage readimage(t_fileName);					//利用文件名构造函数，读取图像
		t_imgO = readimage;								//输出
		return res;
	}

	/*brief:阈值分割
	* param0[i]:输入image
	* param1[o]:输出region
	* param2[i]:最小阈值
	* param3[i]:最大阈值
	*/
	bool OperatorSet::threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = false;
		cv::Mat i_image, o_image;
		cv::threshold(i_image, o_image, t_minGray, t_maxGray,cv::THRESH_BINARY);	//阈值分割
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
		t_reg.GetRegionFeature(0);		//获取最新的region feature
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

		return res;
	}

	bool TestImgProc() {
		bool res = false;

		return res;
	}

}