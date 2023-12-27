#ifndef _DEFECTDETECTION_H_
#define _DEFECTDETECTION_H_

#include "Core.h"
#include "ImgProc.h"

#include <vector>

// 
namespace PZTIMAGE {
	// define enum
	
	enum ImageFlag{IMAGEFLAG_MONO8 = 0, IMAGEFLAG_RGB8, IMAGEFLAG_RGB16};
	enum DefectType{DEFECTTYPE_POINT = 0, DEFECTTYPE_LINE};

	// define struct
	
	// 图像
	typedef struct ImageMeta{
		unsigned int*  m_imgPtr;
		unsigned int 	m_imgW;
		unsigned int 	m_imgH;
		ImageFlag    	m_flag;
	}ImageMeta;
	
	typedef struct FunctionOption{
	
	}FunctionOption;

	typedef struct Defect{	
		DefectType   m_type;
		unsigned int m_area;
	}Defect;

	// brief : 
	// param0: 
	// param1:
	// param2:
	// return: 
	bool MiniLEDPanelDetection(ImageMeta t_imgMeta, FunctionOption t_opt, std::vector<Defect>& t_res);
	
	//dong	分类不做，图像格式16U，功能不设
	bool MiniLEDPanelDetectionV1(ImageMeta t_imgMeta);

	bool TestDefectDetection();
}

#endif
