#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:��ȡͼƬ
	* param0[o]:��ȡ����ͼƬ		����ͨ����ͼ��
	* param1[i]:ͼƬ·��			string
	*/
	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = false;
		PZTImage readimage(t_fileName);					//�����ļ������캯������ȡͼ��
		t_imgO = readimage;								//���
		return res;
	}

	/*brief:��ֵ�ָ�
	* param0[i]:����image			�Ҷ�ͼ��
	* param1[o]:���region			��ֵ��ͼ��0����255
	* param2[i]:��С��ֵ
	* param3[i]:�����ֵ
	*/
	bool OperatorSet::threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = false;
		cv::Mat i_image = t_img.m_image;
		cv::Mat o_image;
		cv::threshold(i_image, o_image, t_minGray, t_maxGray, cv::THRESH_BINARY);	//��ֵ�ָ�ָ����ͨ��ֻ��һ��
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
		t_reg.GetRegionFeature(0);		//��ȡ���µ�region feature�������
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
		t_imgI.ReduceDomain(t_reg);
		t_imgO = t_imgI;
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
			//����������region��ͨ�򣬲��������
			//�������Ҫ���ڵ���ͨ��Ҷ�ֵ��0
			//���⣺��ô���������
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

	/*brief:��ǿͼƬ�Աȶ�
	* param0[i]:����ͼƬ
	* param1[o]:���ͼƬ
	* param2[i]:mask��
	* param3[i]:mask��
	* param4[i]:��ǿǿ��
	*/
	bool OperatorSet::emphasize(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight, uint8_t Factor) {
		bool res = false;
		//��ʽres := round((orig - mean) * Factor) + orig
		//�ȼ�����MaskHeight��MaskWidth�Ŀռ������Ļ������ӷ���
		cv::Mat mean;

		//�ȼ�����ָ����Χ�����ڵľ�ֵ
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
					//����ת�� С��0��ֵ�ᱻ��Ϊ0 ����255��ֵ�ᱻ��Ϊ255
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