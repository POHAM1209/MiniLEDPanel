#ifndef _CORE_H_
#define _CORE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace PZTIMAGE {

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum StructElement { STRUCTELEMENT_RECTANGLE = 0, STRUCTELEMENT_CROSS, STRUCTELEMENT_CIRCLE };

	typedef struct RegionFeature {
		// !
		double 					m_area;
		double 					m_circularity;
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
		PZTImage& operator = (const PZTImage& t_other);
		~PZTImage();

		/* 
		* param0[i]: The path of the image to be read.
		*/
		PZTImage(const std::string& t_filePath);
		
	public:
		/* 
		* brief    : Return the size of an image.
		* param0[o]: The height of an image.
		* param1[o]: The width of an image.
		*/
		bool GetImageSize(unsigned int& t_imgRow, unsigned int& t_imgCol) const;

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
		bool Decompose(cv::Mat& t_ch0, cv::Mat& t_ch1, cv::Mat& t_ch2);

		/* 
		* brief    : Reduce the image with reduced definition domain.
		* param0[i]: New definition domain.
		*/
		bool ReduceDomain(const PZTRegions& t_reg);

		/* 
		* brief    : Transform an RGB image into a gray scale image.
		*/
		bool RGB2Gray();

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
	//		1. connection() return number of connected domain is wrong
	//      2. GetRegionFeature()  findContour is wrong
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTRegions {
		friend bool PZTImage::ReduceDomain(const PZTRegions& t_reg);
	public:
		PZTRegions();
		PZTRegions(const PZTRegions& t_other);
		PZTRegions& operator= (const PZTRegions& t_other);
		~PZTRegions();

		/*
		* param0[i]: labeled region. The grayscale value i represents the i-th connected domain.
		*/
		PZTRegions(cv::Mat t_reg);

	public:
		/* 
		* brief    : Return features of a connected domain.
		* param0[i]: The index of connected domaim. It starts from scratch.
		*/
		RegionFeature GetRegionFeature(unsigned int t_index);

		/*
		* brief    : Compute connected components of a region.
		*/
		bool Connection();

		/*
		* brief    : Gain the union of all connected domains.
		*/
		bool Disconnection();

		bool FillUp();

		/*
		* brief    : Erode a region. By the way, m_regionNum must be 1.
		* param0[i]: Structuring element, such as rectangle, circle.
		* param1[i]: The size of Structuring element, such as 3×3、5×5、7×7.
		*/
		bool Erosion(StructElement t_elm, unsigned int t_kernelLen = 3);

		/*
		* brief    : Get the class member(m_regionNum)
		*/
		unsigned int GetRegionNum();


	private:
		bool _UpdataRegionNum();
		bool _UpdataRegionFeatures();
		bool _UpdataRegions();

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

		// ! the number of connected domains
		unsigned int 									m_regionNum;
	};

	bool TestCore();

}

#endif
