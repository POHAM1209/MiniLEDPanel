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

		/*brief:��ȡͼƬ
		* param0[o]:��ȡ����ͼƬ		����ͨ����ͼ��
		* param1[i]:ͼƬ·��			string
		*/
		static bool read_image(PZTImage& t_imgO, std::string t_fileName);

		/*brief:��ֵ�ָ�
		* param0[i]:����image			�Ҷ�ͼ��
		* param1[o]:���region			��ֵ��ͼ��0����255
		* param2[i]:��С��ֵ
		* param3[i]:�����ֵ
		*/
		static bool threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray);

		/*brief:��ͨ��ָ�
		* param0[i]:����region
		* param1[o]:���region(��ͨ��ֿ�)
		*/
		static bool connection(PZTRegions t_reg, PZTRegions& t_regs);

		/*brief:��image�ϼ�ȥregion
		* param0[i]:����image
		* param1[i]:����region
		* param2[o]:���image
		*/
		static bool reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO);

		/*brief:������״����ѡ����
		* param0[i]:����region��һ��Ϊconnection���region
		* param1[o]:���region
		* param2[i]:������������(��ʲô����ѡ��)
		* param3[i]:��С��Χ
		* param4[i]:���Χ
		*/
		static bool select_shape(PZTRegions t_regI, PZTRegions& t_regO, Features t_fea, float t_min, float t_max);

		/*brief:��ǿͼƬ�Աȶ�
		* param0[i]:����ͼƬ
		* param1[o]:���ͼƬ
		* param2[i]:mask��
		* param3[i]:mask��
		* param4[i]:��ǿǿ��
		*/
		static bool emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor);

		/*brief:
		*
		*/
		static bool rank_rect();

		/*brief:
		*
		*/
		static bool gray_range_rect();

	};

	bool TestImgProc();

}
#endif
