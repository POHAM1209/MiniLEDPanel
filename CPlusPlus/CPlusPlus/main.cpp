#include "Test.h"
#include "Core.h"
#include "ImgProc.h"
#include "DefectDetection.h"

#include <iostream>

enum TestType {
	TESTTYPE_CORE = 0, TESTTYPE_IMGPROC, TESTTYPE_DEFECTDETECTION, TESTTYPE_TEST
};

int main() {

	int type = 1;
	bool res = false;

	switch (type){
		case TESTTYPE_CORE:
			res = PZTIMAGE::TestCore(); break;
		case TESTTYPE_IMGPROC:
			res = PZTIMAGE::TestImgProc(); break;
		case TESTTYPE_DEFECTDETECTION:
			res = PZTIMAGE::TestDefectDetection(); break;
		case TESTTYPE_TEST:
			res = PZTIMAGE::TestTest(); break;
	default:
		break;
	}

	//
	return 0;
}
