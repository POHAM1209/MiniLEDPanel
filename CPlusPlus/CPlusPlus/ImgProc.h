#ifndef _IMGPROC_H_
#define _IMGPROC_H_

#include "Core.h"
#include "Defines.h"

#include <string>

#define PARAM_IN
#define PARAM_OUT
/*------------------------错误记录:enum在namespace内出现枚举重定义错误，将enum移出namespace即可----------------------------*/
//shape_trans所需
enum ShapeTransType
{
	SHAPETRANS_RECTANGLE = 0,	//平行于坐标轴的最小外接矩形
	SHAPETRANS_CIRCLE,		//最小外接圆
	SHAPETRANS_CONVER		//凸包
};

//dyn_threshold所需
enum Light_Dark { Light = 0, Dark, Equal, Not_equal };

namespace PZTIMAGE {
	class OperatorSet {
	public:
		//

		/*brief:读取图片
		* param0[o]:读取到的图片		任意通道数图像
		* param1[i]:图片路径			string
		*/
		static bool read_image(PZTImage& t_imgO, std::string t_fileName);

		/*brief:灰度化
		* param0[i]:输入图片
		* param1[o]:输出图片
		*/
		static bool gray_image(PZTImage t_imgI, PZTImage& t_imgO);

		/*brief:阈值分割
		* param0[i]:输入image			灰度图像
		* param1[o]:输出region			二值化图像，0或者255
		* param2[i]:像素阈值大小，取大于
		* param3[i]:给 大于像素阈值的像素区域 赋值，大于部分均为该值
		*/
		static bool threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_threshold, uint8_t t_maxGray);

		/*brief:连通域分割
		* param0[i]:输入region
		* param1[o]:输出region(连通域分开)
		*/
		static bool connection(PZTRegions t_reg, PZTRegions& t_regs);

		/*brief:在image上减去region
		* param0[i]:输入image
		* param1[i]:输入region
		* param2[o]:输出image
		*/
		static bool reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO);

		/*brief:按照形状特征选择结果
		* param0[i]:输入region，一般为connection后的region
		* param1[o]:输出region
		* param2[i]:输入特征类型(以什么特征选择)
		* param3[i]:最小范围
		* param4[i]:最大范围
		*/
		static bool select_shape(PZTRegions t_regI, PZTRegions& t_regO, Features t_fea, float t_min, float t_max);

		/*brief:增强图片对比度
		* param0[i]:输入图片
		* param1[o]:输出图片
		* param2[i]:mask长
		* param3[i]:mask宽
		* param4[i]:增强强度
		*/
		static bool emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor);

		/*brief:灰度增强
		* param0[i]:输入图片
		* param1[o]:输出图片
		* param2[i]:mask长
		* param3[i]:mask宽
		*/
		static bool gray_range_rect(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight);

		/*brief:取region信息
		* param0[i]:输入region
		* param1[o]:region面积
		* param2[o]:region中心行
		* param3[o]:region中心列
		*/
		static bool area_center(PZTRegions t_regI, int& area, cv::Point2f& point);

		/*brief:填充
		* param0[i]:输入region
		* param1[o]:输出region
		*/
		static bool fill_up(PZTRegions t_regI,PZTRegions t_regO);

		/*brief:圆形腐蚀
		* param0[i]:输入region
		* param1[o]:输出region
		* param2[i]:圆形半径
		*/
		static bool erosion_circle(PZTRegions t_regI, PZTRegions& t_regO, int t_radius);

		/*brief:中值滤波
		* param0[i]:输入图片
		* param1[o]:输出图片
		* param2[i]:mask长
		* param3[i]:mask宽
		*/
		static bool mean_image(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight);

		/*brief:动态阈值分割
		* param0[i]:输入图片
		* param1[i]:输入阈值图片，如mean_image处理后的图片
		* param2[o]:输出分割图片
		* param3[i]:offset值 补偿，一般设为5-40之间
		* param4[i]:分割结果条件选择，默认Light为大于阈值图片
		*/
		static bool dyn_threshold(PZTImage t_imgI, PZTImage t_thresholdimgI, PZTRegions& t_regO, uint8_t t_offset, Light_Dark Light_Dark);

		static bool display_image(PZTImage t_imgI, std::string WindowName,bool save);

		static bool display_region(PZTRegions t_regI, float multiple);

		static bool opening_rectangle1(PZTRegions t_regI, PZTRegions& t_regO, uint8_t t_Width, uint8_t t_Height);
		static bool opening_circle(PZTRegions t_regI, PZTRegions& t_regO, uint8_t radius);
		static bool dilation_rectangle1(PZTRegions t_regI, PZTRegions& t_regO, uint8_t t_Width, uint8_t t_Height);

		static bool move_region(PZTRegions t_regI, PZTRegions& t_regO, int rows, int cols);

		static bool union2(PZTRegions t_regI1, PZTRegions t_regI2 ,PZTRegions& t_regO);
		static bool union1(PZTRegions t_regI, PZTRegions& t_regO);
		static bool shape_trans(PZTRegions t_regI, PZTRegions& t_regO, ShapeTransType t_type);

		static bool difference(PZTRegions t_regI1, PZTRegions t_regI2, PZTRegions& t_regO);

		static bool region_features(PZTRegions t_regI, Features t_fea, int& Value);
		/*---------------------------------------新增----------------------------------------*/
		/*brief:
		*
		*/
		
		
		//region取反
		static bool complement(PZTRegions t_regI, PZTRegions& t_regO);
		//region叠加,是否需要image与region叠加,目前不需要
		static bool concat_region(PZTRegions t_regI, PZTRegions& t_regO);
		//region之间的交集
		static bool intersection(PZTRegions t_regI1, PZTRegions t_regI2, PZTRegions& t_regO);
	};

	bool TestImgProc();

}
#endif
