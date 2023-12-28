#include "DefectDetection.h"

namespace PZTIMAGE {

	bool MiniLEDPanelDetectionV1(ImageMeta t_imgMeta)
	{
		bool res = true;
		
		//flag是否应该修改
		if (t_imgMeta.m_flag != IMAGEFLAG_RGB16)
			return false;
		//指针数据转成Mat
		cv::Mat image(t_imgMeta.m_imgW, t_imgMeta.m_imgH,CV_8UC3 ,t_imgMeta.m_imgPtr);	//8UC3，还是其它
		cv::Mat mask= cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(1));
		PZTImage Image(image,mask);

		uint8_t chipGrayValue = 170;			//检测芯片所需最小灰度值，绿色背景图170，灰色背景图200
		uint8_t backGrayValue = 120;			//检测背景所需最大灰度值
		uint16_t chipminArea = 4500;			//芯片面积(绿色背景图 5000附近)
		uint16_t chipmaxArea = 6000;
		unsigned int AOIWidth = 150;			//绿色背景芯片宽(横向)
		unsigned int AOIHeight = 200;			//绿色背景芯片高(纵向)
		uint16_t rectMinArea = 4500;			//方框的最小面积，中间镂空，数值等于膨胀宽高的周长700？由于计算面积的方式，计算的是整体面积，则是5000
		unsigned int width, height;

		//图像部分
		PZTImage Image, Gray, R, G, B, Emphasize, Reduce, Mean;
		PZTRegions chipRegion, Background, Connection, chipSelectRegion, Union, Opening,
			Trans, Dyn, Dilation, resultSelectRegion, Connection1, Opening1;
		//算子部分
		OperatorSet::get_image_size(Image, width, height);
		OperatorSet::decompose3(Image, R, G, B);		//当图片已为灰度图时
		OperatorSet::emphasize(B, Emphasize, 36, 36, 3);
		OperatorSet::threshold(Emphasize, chipRegion, chipGrayValue, 255);			//检测芯片用
		//OperatorSet::threshold(Emphasize, Background, 0, backGrayValue);			//检测背景用
		OperatorSet::opening_rectangle1(chipRegion, Opening, 10, 10);				//使用opening运行速度快了，加速的原因是跳过updata了?
		OperatorSet::connection(Opening, Connection);
		OperatorSet::select_shape(Connection, chipSelectRegion, FEATURES_AREA, chipminArea, chipmaxArea);		//会存在芯片计算错误
		OperatorSet::shape_trans(chipSelectRegion, Trans, SHAPETRANSTYPE_RECTANGLE1);
		OperatorSet::union1(Trans, Union);
		OperatorSet::dilation_rectangle1(Union, Dilation, AOIWidth, AOIHeight);			//膨胀到比芯片略大
		OperatorSet::reduce_domain(B, Dilation, Reduce);
		OperatorSet::mean_image(Reduce, Mean, 7, 7);
		OperatorSet::dyn_threshold(Reduce, Mean, Dyn, 7, Dark);							//去除芯片后会留下方形边缘

		//将方框去掉，耗时翻倍
		OperatorSet::connection(Dyn, Dyn);
		OperatorSet::select_shape(Dyn, resultSelectRegion, FEATURES_AREA, 0, rectMinArea);	//将方框去掉，只选缺陷
		resultSelectRegion.DisplayRegion();

		return res;
	}

	bool MiniLEDPanelDetection(ImageMeta t_imgMeta, FunctionOption t_opt, std::vector<Defect>& t_res){
		bool res = false;
		t_res.clear();

		return res;
	}

	bool TestDefectDetection() {
		bool res = false;

		return res;
	}

}
