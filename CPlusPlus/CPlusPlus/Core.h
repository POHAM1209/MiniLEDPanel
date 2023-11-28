#ifndef _CORE_H_
#define _CORE_H_

#include "Defines.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

namespace PZTIMAGE {

	#define APPROXIMATION_ACCURACY		3

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum StructElement { STRUCTELEMENT_RECTANGLE = 0, STRUCTELEMENT_CROSS, STRUCTELEMENT_CIRCLE };

	typedef struct RegionFeature {
		// ! 考虑内存对齐
		float 					m_area;
		float 					m_circularity;
		unsigned int 			m_row;
		unsigned int 			m_col;
		float					m_outerRadius;
		float					m_innerRadius;
	}RegionFeature;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// m_image is CV_8U; m_mask is CV_8UC1 
	//
	// m_mask copyTo
	// cv::Mat::refcount refers to the shared memory.   
	// PZTImage a = b; --> they(a, b) refer to the same memory.
	// existing problems: mult_image() do not think about float
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTRegions;

	class PZTImage {
		friend class OperatorSet;
	public:
		PZTImage();
		PZTImage(const PZTImage& t_other);
		PZTImage(PZTImage&& t_other);
		PZTImage& operator = (const PZTImage& t_other);
		PZTImage& operator = (PZTImage&& t_other);
 		~PZTImage();

		/* 
		* param0[i]: The path of the image to be read.
		*/
		PZTImage(const std::string& t_filePath);

		/* 
		* param0[i]: the input image.
		* param1[i]: the input mark. The default mask is the whole image.
		*/
		PZTImage(const cv::Mat& t_img, const cv::Mat& t_mask = cv::Mat());
		PZTImage(const cv::Mat& t_img, cv::Mat&& t_mask = cv::Mat());

		/*
		* no practical purpose!!!! 只处理两个图片相交区域
		*/
		PZTImage operator - (const PZTImage& t_other) const;
		PZTImage operator + (const PZTImage& t_other) const;
		PZTImage operator - (uint8_t t_val) const;
		PZTImage operator + (uint8_t t_val) const;
		
	public:
		/* 
		* brief    : Determine whether the object is empty.
		*/
		bool Empty() const;

		/*
		* brief    : Creates a full copy.
		*/
		PZTImage Clone() const;

		/* 
		* brief    : Return the number of image channels.
		*/
		int Channels() const;

		/* 
		* brief    : Return the size of an image.
		* param0[o]: The height of an image.
		* param1[o]: The width of an image.
		*/
		bool GetImageSize(unsigned int& t_imgRow, unsigned int& t_imgCol) const;

		/* 
		* brief    : Return the size of an image.
		* param0[o]: The size(width/col, height/row) of an image.
		*/
		bool GetImageSize(cv::Size2i& t_size) const;

		/* 
		* brief    : Convert 3 images with one channel into a three-channel image.
		* param0[i]: The 1st channel.
		* param1[i]: The 2nd channel.
		* param2[i]: The 3rd channel.
		*/
		bool Compose(const cv::Mat& t_ch0, const cv::Mat& t_ch1, const cv::Mat& t_ch2);

		/* 
		* brief    : Convert a three-channel image into three images with one channel.
		* param0[o]: The 1st channel.
		* param1[o]: The 2nd channel.
		* param2[o]: The 3rd channel.
		*/
		bool Decompose(cv::Mat& t_ch0, cv::Mat& t_ch1, cv::Mat& t_ch2) const;

		/* 
		* brief    : Convert a three-channel image into three images with one channel.
		* param0[o]: The 1st channel.
		* param1[o]: The 2nd channel.
		* param2[o]: The 3rd channel.
		*/
		bool Decompose(PZTImage& t_ch0, PZTImage& t_ch1, PZTImage& t_ch2) const;

		/* 
		* brief    : Smooth by averaging.
		* param0[i]: Width of filter mask.
		* param1[i]: Height of filter mask.
		*/
		bool Mean(uint32_t t_maskWidth, uint32_t t_maskHeight);

		/* 
		* brief    : Segment an image with one channel using global threshold.
		* param0[o]: The output region meeting the condition.
		* param1[i]: Lower threshold for the gray values.
		* param2[i]: Upper threshold for the gray values.
		*/
		bool Threshold(PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray) const;

		/* 
		* brief    : Reduce the image with reduced definition domain.
		* param0[i]: New definition domain.
		*/
		bool ReduceDomain(const PZTRegions& t_reg);

		/* 
		* brief    : Transform an RGB image into a gray scale image.
		*/
		bool RGB2Gray();

		/* 
		* brief    : Transform an image from an color space to another color space.
		* test     : to do.
		*/
		bool ChangeColorSpace(TransColorSpace t_color);

		void DisplayImage(float t_factor = 1);

	private:

	private:
		cv::Mat 										m_image;
		cv::Mat 										m_mask;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	// The flags of input array in cv::threshold and cv::connectedComponents is CV_8U
	//
	// the image processing operator related with Halcon Region includes 
	// 	-- threshold(region)
	//	-- erosion_xxx/dilation_xxx(region array)
	//	-- select_shape(region array)
	//	-- union1(region array)
	//	-- union2(region array)
	// 
	//  Q:
	//		1. m_featuresPtr在特殊情况报异常   m_featurePtr初始化nullptr，但调用GetRegionFeature()，再创建指针。
	//   	当 m_featurePtr为非 nullptr 的对象构造/赋值，感觉需要额外的成员变量记录是否被修改。
	//
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTRegions {
		friend bool PZTImage::ReduceDomain(const PZTRegions& t_reg);
	public:
		PZTRegions();
		PZTRegions(const PZTRegions& t_other);
		PZTRegions(PZTRegions&& t_other);
		PZTRegions& operator= (const PZTRegions& t_other);
		~PZTRegions();

		/*
		* param0[i]: labeled region. The grayscale value i represents the i-th connected domain.
		*/
		PZTRegions(const cv::Mat& t_reg);
		PZTRegions(cv::Mat&& t_reg);

		/*
		* param0[i]: The reference template.
		* param1[i]: The index set of region from the input of PZTRegion object.
		*/
		PZTRegions(const PZTRegions& t_reg, const std::vector<uint32_t>& t_indexs);


		PZTRegions operator + (const PZTRegions& t_other) const;

	public:
		/* 
		* brief    : Clear all connected domains, but still ramain memory from m_regions.
		*/
		bool Clear();

		/* 
		* brief    : Determine whether the object is empty.
		*/
		bool Empty() const;

		/*
		* brief    : Get the class member(m_regionNum)
		*/
		unsigned int GetRegionNum() const;

		/* 
		* brief    : Return the size of container storing the detail of regions.
		* param0[o]: The size(width/col, height/row) of the container.
		*/
		bool GetRegionSize(cv::Size2i& t_size) const;

		/* 
		* brief    : Return features of a connected domain.
		* param0[i]: The index of connected domaim. It starts from scratch.
		*/
		RegionFeature GetRegionFeature(unsigned int t_index);

		/* 
		* brief    : Return specific feature of a connected domain.
		* param0[i]: The index of connected domaim. It starts from scratch.
		* param1[i]: Shape features to be checked.
		* return   : The value of feature.
		*/
		float GetRegionFeature(unsigned int t_index, FeatureType t_type);

		/*
		* brief    : Translate a region.
		* param0[i]: Row coordinate of the vector by which the region is to be moved.
		* param1[i]: Col coordinate of the vector by which the region is to be moved.
		*/
		bool MoveRegion(int t_row, int t_col);

		/*
		* brief    : Compute connected components of a region.
		*/
		bool Connection();

		/*
		* brief    : Gain the union of all connected domains.
		*/
		bool Disconnection();

		/*
		* brief    : Fill up holes in one region. *** m_regionNum must be 1 ***
		*/
		bool FillUp();

		/*
		* brief    : Transform the shape of a region.  目前只实现 rectangle1
		* param0[i]: Type of transformation.
		*/
		bool ShapeTrans(ShapeTransType t_type);

		/*
		* brief    : Morphological processing. *** m_regionNum must be 1 ***
		* param0[i]: Structuring element, such as rectangle, circle.
		* param1[i]: The size of Structuring element, such as 3×3、5×5、7×7.
		*/
		bool Erosion(StructElement t_elm, unsigned int t_kernelWidth = 3, unsigned int t_kernelHeight = 3);
		bool Dilation(StructElement t_elm, unsigned int t_kernelWidth = 3, unsigned int t_kernelHeight = 3);
		bool Opening(StructElement t_elm, unsigned int t_kernelWidth = 3, unsigned int t_kernelHeight = 3);


		void DisplayRegion(float t_factor = 1);

	private:
		bool _UpdataRegionNum();
		bool _UpdataRegionFeatures();
		bool _UpdataRegions();


		bool _UpdataRegionsFeaturesV2();
		RegionFeature _GainOneRegionFeatures(cv::InputArray t_oneRegion);


	private:
		/* ! The data container. There are followed details.
			   -- 1 channel, 8 bit. (equals to CV_8UC1)
			   -- The grayscale value from the pixel in the i-th connected domain equals to i, 
			      and value '0' represents background.
		*/
		cv::Mat											m_regions;

		/* ! pointer to the features. There are followed details. 
			   -- 
		*/
		std::shared_ptr<std::vector<RegionFeature>>		m_featuresPtr;

		// ! the number of connected domains.
		unsigned int 									m_regionNum;

		// ! It represents whether m_regions has changed by some operators
		bool											m_isRegionChanged;
	};

	bool TestCore();

}

#endif
