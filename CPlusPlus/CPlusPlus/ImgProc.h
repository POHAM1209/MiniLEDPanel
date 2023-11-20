#ifndef _IMGPROC_H_
#define _IMGPROC_H_

#include "Core.h"
#include "Defines.h"

#include <string>

#define PARAM_IN
#define PARAM_OUT

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
		* param2[i]:最小阈值
		* param3[i]:最大阈值
		*/
		static bool threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray);

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

		/*brief:增强图片对比度				灰度图
		* param0[i]:输入图片
		* param1[o]:输出图片
		* param2[i]:mask长
		* param3[i]:mask宽
		* param4[i]:增强强度
		*/
		static bool emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor);

		/*brief:灰度增强					灰度图
		* param0[i]:输入图片
		* param1[o]:输出图片
		* param2[i]:mask长
		* param3[i]:mask宽
		*/
		static bool gray_range_rect(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight);

	};

	bool TestImgProc();

}
#endif
