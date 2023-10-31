#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:��ȡͼƬ
	* param0[o]:��ȡ����ͼƬ
	* param1[i]:ͼƬ·��
	*/
	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = false;
		PZTImage readimage(t_fileName);					//�����ļ������캯������ȡͼ��
		t_imgO = readimage;								//���
		return res;
	}

	/*brief:��ֵ�ָ�
	* param0[i]:����image
	* param1[o]:���region
	* param2[i]:��С��ֵ
	* param3[i]:�����ֵ
	*/
	bool OperatorSet::threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = false;
		cv::Mat i_image, o_image;
		cv::threshold(i_image, o_image, t_minGray, t_maxGray,cv::THRESH_BINARY);	//��ֵ�ָ�
		PZTRegions n_region(o_image);												//����ͼ����region���õ���ֵ�ָ���ͼ��
		t_reg = n_region;															//���
		return res;
	}

	/*brief:��ͨ��ָ�
	* param0[i]:����region
	* param1[o]:���region(��ͨ��ֿ�)
	*/
	bool OperatorSet::connection(PZTRegions t_reg, PZTRegions& t_regs) {
		bool res = false;
		t_reg.Connection();				//��ͨ��ָ�
		t_reg.GetRegionFeature(0);		//��ȡ���µ�region feature
		t_regs = t_reg;					//���
		return res;
	}

	/*brief:��image�ϼ�ȥregion
	* param0[i]:����image
	* param1[i]:����region
	* param2[o]:���image
	*/
	bool OperatorSet::reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO) {
		bool res = false;
		//����������image������ֱ��container����������Ǿ������

		return res;
	}

	/*brief:������״����ѡ����
	* param0[i]:����region��һ��Ϊconnection���region
	* param1[o]:���region
	* param2[i]:������������(��ʲô����ѡ��)
	* param3[i]:��С��Χ
	* param4[i]:���Χ
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