#ifndef _DEFECTDETECTION_H_
#define _DEFECTDETECTION_H_

#define IntPtr long long*

// 
namespace PZTIMAGE {
	// define enum
	
	enum ImageFlag{IMAGEFLAG_MONO8 = 0, IMAGEFLAG_RGB8};
	enum DefectType{DEFECTTYPE_POINT = 0, DEFECTTYPE_LINE};

	// define struct
	
	typedef struct ImageMeta{
		IntPtr       m_imgPtr;
		unsigned int m_imgW;
		unsigned int m_imgH;
		ImageFlag    m_flag;
	}ImageMeta;
	
	typedef struct FunctionOption{
	
	}FunctionOption;

	typedef struct Defect{
	
		DefectType   m_type;
		unsigned int m_area;
	}Defect;

	// brief : 
	// param0: 
	// return: 
	bool MiniLEDPanelDetection(IntPtr t_imgPtr, ImageMeta t_imgMeta, FunctionOption t_opt, std::vector<Defect>& t_res);
	
	bool TestDefectDetection();
}

#endif
