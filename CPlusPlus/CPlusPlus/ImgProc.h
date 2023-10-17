#ifndef _IMGPROC_H_
#define _IMGPROC_H_

#include "Core.h"

namespace PZTIMAGE {

	class OPeratorSet{
	public:
		// 
		static bool threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray); 

		//
		static bool connection(PZTRegions t_reg, PZTRegions& t_regs);

		//
		static bool  
	};

	bool TestImgProc();

}
#endif
