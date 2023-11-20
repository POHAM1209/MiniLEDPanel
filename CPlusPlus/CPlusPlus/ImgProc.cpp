#include "ImgProc.h"

namespace PZTIMAGE {

	/*brief:��ȡͼƬ
	* param0[o]:��ȡ����ͼƬ		����ͨ����ͼ��
	* param1[i]:ͼƬ·��			string
	*/
	bool OperatorSet::read_image(PZTImage& t_imgO, std::string t_fileName) {
		bool res = true;
		if (t_fileName.empty())
			return false;
		PZTImage readimage(t_fileName);					//�����ļ������캯������ȡͼ��
		t_imgO = readimage;								//���
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

	/*brief:��ֵ�ָ�
	* param0[i]:����image			�Ҷ�ͼ��
	* param1[o]:���region			��ֵ��ͼ��0����255
	* param2[i]:��С��ֵ
	* param3[i]:�����ֵ
	*/
	bool OperatorSet::threshold(PZTImage t_imgI, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		cv::Mat i_image = t_imgI.m_mask;
		cv::Mat o_image;
		cv::threshold(i_image, o_image, t_minGray, t_maxGray, cv::THRESH_BINARY);	//��ֵ�ָ�ָ����ͨ��ֻ��һ��
		PZTRegions o_region(o_image);												//����ͼ����region���õ���ֵ�ָ���ͼ��
		t_reg = o_region;															//���
		
		return res;
	}

	/*brief:��ͨ��ָ�
	* param0[i]:����region
	* param1[o]:���region(��ͨ��ֿ�)
	*/
	bool OperatorSet::connection(PZTRegions t_reg, PZTRegions& t_regs) {
		bool res = true;

		t_reg.Connection();				//��ͨ��ָ�ѵõ�m_regionNum,m_regions��ȱ��m_feature
		for (int i = 0; i < t_reg.GetRegionNum(); i++)
		{
			//��Ҫ���������գ�����ֱ���ں��������? 
			t_reg.GetRegionFeature(i);
		}
		t_regs = t_reg;					//���
		return res;
	}

	/*brief:��image�ϼ�ȥregion
	* param0[i]:����image
	* param1[i]:����region
	* param2[o]:���image
	*/
	bool OperatorSet::reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage& t_imgO) {
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

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
		bool res = true;

		//������region���жϿ�
		//if (t_regI.m_regions.empty())
		//	return false;

		int num = t_regI.GetRegionNum();
		if (t_fea == FEATURES_AREA)
		{
			//����������region��ͨ�򣬲��������
			//���⣺�������Ҫ���ڵ���ͨ��Ҷ�ֵ��0
			for (int i = 1; i < num; i++)//0�Ǳ������Ǵ�0���Ǵ�1��ʼ����
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_area >= t_min && regf.m_area <= t_max)
					continue;
				else
				{
					//��ν������ϵĻҶ�ֵ��0? ��ô�õ�regionλ������0?

				}
			}
		}
		else if (t_fea == FEATURES_CIRCULARITY)
		{
			for (int i = 1; i < num; i++)//0�Ǳ������Ǵ�0���Ǵ�1��ʼ����
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_circularity >= t_min && regf.m_circularity <= t_max)
					continue;
				else
				{
					//��ν������ϵĻҶ�ֵ��0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else if (t_fea == FEATURES_ROW)
		{
			for (int i = 1; i < num; i++)//0�Ǳ������Ǵ�0���Ǵ�1��ʼ����
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_row >= t_min && regf.m_row <= t_max)
					continue;
				else
				{
					//��ν������ϵĻҶ�ֵ��0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else if (t_fea == FEATURES_COLUMN)
		{
			for (int i = 1; i < num; i++)//0�Ǳ������Ǵ�0���Ǵ�1��ʼ����
			{
				RegionFeature regf = t_regI.GetRegionFeature(i);
				if (regf.m_col >= t_min && regf.m_col <= t_max)
					continue;
				else
				{
					//��ν������ϵĻҶ�ֵ��0? how to set the value of unmatched pixel to 0; 

				}
			}
		}
		else
		{
			std::cout << "error! there is not this Feature!" << std::endl;
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
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		//��ʽres := round((orig - mean) * Factor) + orig
		//�ȼ�����MaskHeight��MaskWidth�Ŀռ������Ļ������ӷ���
		cv::Mat mean;

		//�Ե�ͨ��Ϊ׼��m_mask
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

	/*brief:�Ҷ���ǿ
	* param0[i]:����ͼƬ
	* param1[o]:���ͼƬ
	* param2[i]:mask��
	* param3[i]:mask��
	*/
	bool OperatorSet::gray_range_rect(PZTImage t_imgI, PZTImage& t_imgO, uint8_t t_MaskWidth, uint8_t t_MaskHeight)
	{
		bool res = true;

		if (t_imgI.m_image.empty())
			return false;
		if (t_imgI.m_image.type() == CV_8UC3)
			return false;

		//ͼ��߽�����
		int hh = (t_MaskHeight - 1) / 2;
		int hw = (t_MaskWidth - 1) / 2;
		cv::Mat Newsrc;
		cv::copyMakeBorder(t_imgI.m_mask, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//�Ա�ԵΪ�ᣬ�Գ�
		t_imgO.m_mask= cv::Mat::zeros(t_imgI.m_mask.rows, t_imgI.m_mask.cols, t_imgI.m_mask.type());

		//����ͼ��
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