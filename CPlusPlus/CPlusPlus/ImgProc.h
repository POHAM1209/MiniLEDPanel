#ifndef _IMGPROC_H_
#define _IMGPROC_H_

#include "Core.h"
#include "Defines.h"

#include <string>

#define PARAM_IN
#define PARAM_OUT

namespace PZTIMAGE {

	class OperatorSet{
	public:
		//

		/*
		* brief :
		* param[o]:
		* param[i]:
		* return :
		*/
		static bool read_image(PZTImage& t_imgO, std::string t_fileName);

		// 
		static bool threshold(PZTImage t_img, PZTRegions& t_reg, uint8_t t_minGray, uint8_t t_maxGray); 

		//
		static bool connection(PZTRegions t_reg, PZTRegions& t_regs);

		//
		static bool reduce_domain(PZTImage t_imgI, PZTRegions t_reg, PZTImage t_imgO);
	
		//
		static bool select_shape(PZTRegions t_regI, PZTRegions t_regO, Features t_fea, float t_min, float t_max); 

	};

	bool TestImgProc();

}
#endif
