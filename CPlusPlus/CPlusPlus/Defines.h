#ifndef _DEFINES_H_
#define _DEFINES_H_

namespace PZTIMAGE{

    enum Features{
    	FEATURES_AREA = 0,
		FEATURES_CIRCULARITY,
		FEATURES_ROW,
		FEATURES_COLUMN
    };


	///////////////////////////////////////////////////////////
	// POMAH

	//
	enum FeatureType {FEATURETYPE_AREA = 0, FEATURETYPE_CIRCULARITY, FEATURETYPE_ROW, FEATURETYPE_COL};

	/*
	* Extract light, dark or similar areas? It is used for the image operator dyn_threshold().
	*/ 
	enum PM_LightDark {LIGHTDARK_LIGHT = 0, LIGHTDARK_DARK};

	/*
	* Bayer image format used for the image operator, such as cfa_to_rgb().
	*/ 
	enum BayerType {BAYERTYPE_GB = 0, BAYERTYPE_BG, BAYERTYPE_GR, BAYERTYPE_RG};

	/*
	* Type of transformation used for the image operator shape_trans().
	*/
	enum ShapeTransType {SHAPETRANSTYPE_RECTANGLE1 = 0}; //, SHAPETRANSTYPE_INNER_RECTANGLE1};

	enum TransColorSpace {
		TRANSCOLORSPACE_UNKNOW = 0,
		TRANSCOLORSPACE_RGB2GRAY, 
		TRANSCOLORSPACE_BayerRGGB2RGB, 
		TRANSCOLORSPACE_BayerBG2RGB,
		TRANSCOLORSPACE_BayerGR2RGB,
		TRANSCOLORSPACE_BayerRG2RGB,
		TRANSCOLORSPACE_BayerGB2RGB};

	// Dong

	//dyn_threshold所需
	enum Light_Dark { Light = 0, Dark, Equal, Not_equal };

}


#endif
