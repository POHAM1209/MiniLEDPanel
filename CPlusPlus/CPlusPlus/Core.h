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

	class PZTRegions;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum StructElement { STRUCTELEMENT_CIRCLE, STRUCTELEMENT_RECTANGLE };

	typedef struct RegionFeature {
		unsigned int 			m_area;
		unsigned int 			m_row;
		unsigned int 			m_column;

	}RegionFeature;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// m_image is CV_8U; m_mask is CV_8UC1 
	//
	// m_mask copyTo
	// cv::Mat::refcount refers to the shared memory.   
	// PZTImage a = b; --> they(a, b) refer to the same memory.
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
		bool GetImageSize(unsigned int& t_imgRow, unsigned int& t_imgCol);

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
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTRegions {
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
		* param0[i]: The index of connected domain.
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

		/*
		* brief    : Erode a region
		* param0[i]: Structuring element, such as rectangle, circle.
		*/
		bool Erosion(StructElement t_elm);

	private:
		bool _UpdataRegionNum();
		bool _UpdataRegionFeatures();
		bool _UpdataRegions();

	private:
		// Mat::flags is CV_8UC1
		cv::Mat											m_regions;
		std::shared_ptr<std::vector<RegionFeature>>		m_featuresPtr;
		unsigned int 									m_regionNum;
	};

	bool TestCore();

}

#endif
