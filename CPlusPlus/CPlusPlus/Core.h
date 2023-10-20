#ifndef _CORE_H_
#define _CORE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <string>
#include <vector>

namespace PZTIMAGE{

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	typedef struct RegionFeature{
		unsigned int 			m_area;
		unsigned int 			m_row;
		unsigned int 			m_column;
		
	}RegionFeature;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTImage{
	public:
		PZTImage();
		PZTImage(const std::string& t_fileName);
		PZTImage(const PZTImage& t_other);
		PZTImage& operator = (const PZTImage& t_other);
		~PZTImage();

	private:
		cv::Mat m_container;
		cv::Mat m_mark;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	class PZTRegions{
	public:
		PZTRegions();
		PZTRegions(cv::InputArray t_regI);
		PZTRegions(const PZTRegions& t_other);
		PZTRegions& operator= (const PZTRegions& t_other);
		~PZTRegions();

	public:
		RegionFeature GetRegionFeature(unsigned int t_index);

		bool Connection();

	private:
		bool _UpdataRegionNum();
		bool _UpdataRegionFeatures();
	
	private:
		// CV_16U
		cv::Mat      					m_container;
		//
		unsigned int 					m_regNum;
		// 
		std::vector<RegionFeature> 		m_features;
	};

	bool TestCore();

}

#endif
