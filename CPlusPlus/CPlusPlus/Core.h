#ifndef _CORE_H_
#define _CORE_H_

#include "Test.h"
#include "Utils.h"
#include "Defines.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/ocl.hpp>

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

// ! The number of thread in the ThreadPool.
#define DEFAULT_THREAD_NUM												8

// ! Enable multi-thread acceleration.
//#define HAVE_MULTITHREAD_ACCELERATION

// ! The depth of m_regions, the member of PZTRegion, is CV_8U.
//#define PZTREGION_MAT_8U

// ! The depth of m_regions, the member of PZTRegion, is CV_16S.
#ifndef PZTREGION_MAT_8U
	#define PZTREGION_MAT_16U
#endif

// ! The type of m_regions which is the member of Class PZTRegion.
#ifdef PZTREGION_MAT_8U
	#define PZTREGION_M_REGIONS_TYPE									CV_8UC1
#endif
#ifdef PZTREGION_MAT_16U
	#define PZTREGION_M_REGIONS_TYPE									CV_16UC1
#endif

/*
* Test cases:
*		1. [ ] reduce_domain() 可视化测试。目前：：
*		2. [ ] select_shape() 选取对象有误（提取特征值有误）。目前：area看起来可以
* 		3. [ ] connection() 调用后只能存256个区域。
*		4. [ ] 补充算子 closing_circle(√)/complement(√)/concat_region()/intersection(√)/
*	2023-12-15
*		1. [√] PZTImage::Display()需要添加RGB展示
* 		2. [√] 添加矩阵度
*	2023-12-18		
*		1. [√] 解决最多存储255缺陷()
* 	2023-12-22
*		1. [ ] findCountour();
*/

namespace PZTIMAGE {

	#define APPROXIMATION_ACCURACY		3

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 		
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	enum StructElement { STRUCTELEMENT_RECTANGLE = 0, STRUCTELEMENT_CROSS, STRUCTELEMENT_CIRCLE };

	typedef struct RegionFeature {
		// ! 考虑内存对齐

		// ! Area of the object
		float 					m_area;
		// ! Total length of extreme outer contour
		float					m_contlength;
		// ! The circularity of extreme outer contour
		float 					m_circularity;
		// ! The rectangularity of extreme outer contour
		float 					m_rectangularity;

		// ! Row index of the center
		unsigned int 			m_row;
		// ! Column index of the center
		unsigned int 			m_col;

		float					m_outerRadius;
		float					m_innerRadius;

		// ! Width of the region (parallel to the coordinate axes)
		float					m_width1;
		// ! Height of the region (parallel to the coordinate axes)
		float					m_height1;
		// ! Ratio of the height and the width of the region (parallel to the coordinate axes) = m_height1 / m_width1
		float 					m_ratio1;
		// ! Width of the rotated rectangle obtained through region
		float					m_width2;
		// ! Height of the rotated rectangle obtained through region
		float					m_height2;
		// ! Ratio of the height and the width of the region (rotated rectangle) = m_height2 / m_width2
		float 					m_ratio2;
	}RegionFeature;

	typedef struct NoOp{
    NoOp(){}
		inline void init(int /*labels*/){}
		inline void initElement(const int /*nlabels*/){}

		inline void operator()(int r, int c, int l){
			CV_UNUSED(r);
			CV_UNUSED(c);
			CV_UNUSED(l);
		}

		void finish(){}

		inline void setNextLoc(const int /*nextLoc*/){}
		inline static void mergeStats(const cv::Mat& /*imgLabels*/, NoOp * /*sopArray*/, NoOp& /*sop*/, const int& /*nLabels*/){}
	}NoOp;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	// 		This
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	template<typename LabelT>
	inline static LabelT flattenL(LabelT* P, LabelT length) {
		LabelT k = 1;
		for (LabelT i = 1; i < length; ++i) {
			if (P[i] < i) {
				P[i] = P[P[i]];
			}
			else {
				P[i] = k; k = k + 1;
			}
		}
		return k;
	}

	template<typename LabelT>
	inline static LabelT findRoot(const LabelT* P, LabelT i) {
		LabelT root = i;
		while (P[root] < root) {
			root = P[root];
		}
		return root;
	}

	template<typename LabelT>
	inline static void setRoot(LabelT* P, LabelT i, LabelT root) {
		while (P[i] < i) {
			LabelT j = P[i];
			P[i] = root;
			i = j;
		}
		P[i] = root;
	}

	template<typename LabelT> inline static
	LabelT set_union(LabelT* P, LabelT i, LabelT j) {
		LabelT root = findRoot(P, i);
		if (i != j) {
			LabelT rootj = findRoot(P, j);
			if (root > rootj) {
				root = rootj;
			}
			setRoot(P, j, root);
		}
		setRoot(P, i, root);
		return root;
	}

	template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
	struct LabelingBolelli
	{
		LabelT operator()(const cv::Mat& img, cv::Mat& imgLabels, int connectivity, StatsOp& sop)
		{
			CV_Assert(img.rows == imgLabels.rows);
			CV_Assert(img.cols == imgLabels.cols);
			CV_Assert(connectivity == 8);

			const int h = img.rows;
			const int w = img.cols;

			const int e_rows = h & -2;
			const bool o_rows = h % 2 == 1;
			const int e_cols = w & -2;
			const bool o_cols = w % 2 == 1;

			// A quick and dirty upper bound for the maximum number of labels.
			// Following formula comes from the fact that a 2x2 block in 8-connectivity case
			// can never have more than 1 new label and 1 label for background.
			// Worst case image example pattern:
			// 1 0 1 0 1...
			// 0 0 0 0 0...
			// 1 0 1 0 1...
			// ............
			const size_t Plength = size_t(((h + 1) / 2) * size_t((w + 1) / 2)) + 1;

			std::vector<LabelT> P_(Plength, 0);
			LabelT *P = P_.data();
			//P[0] = 0;
			LabelT lunique = 1;

			// First scan

			// We work with 2x2 blocks
			// +-+-+-+
			// |P|Q|R|
			// +-+-+-+
			// |S|X|
			// +-+-+

			// The pixels are named as follows
			// +---+---+---+
			// |a b|c d|e f|
			// |g h|i j|k l|
			// +---+---+---+
			// |m n|o p|
			// |q r|s t|
			// +---+---+

			// Pixels a, f, l, q are not needed, since we need to understand the
			// the connectivity between these blocks and those pixels only matter
			// when considering the outer connectivities

			// A bunch of defines is used to check if the pixels are foreground
			// and to define actions to be performed on blocks
			{
				#define CONDITION_B img_row_prev_prev[c-1]>0
				#define CONDITION_C img_row_prev_prev[c]>0
				#define CONDITION_D img_row_prev_prev[c+1]>0
				#define CONDITION_E img_row_prev_prev[c+2]>0

				#define CONDITION_G img_row_prev[c-2]>0
				#define CONDITION_H img_row_prev[c-1]>0
				#define CONDITION_I img_row_prev[c]>0
				#define CONDITION_J img_row_prev[c+1]>0
				#define CONDITION_K img_row_prev[c+2]>0

				#define CONDITION_M img_row[c-2]>0
				#define CONDITION_N img_row[c-1]>0
				#define CONDITION_O img_row[c]>0
				#define CONDITION_P img_row[c+1]>0

				#define CONDITION_R img_row_fol[c-1]>0
				#define CONDITION_S img_row_fol[c]>0
				#define CONDITION_T img_row_fol[c+1]>0

				// Action 1: No action
				#define ACTION_1 img_labels_row[c] = 0;
				// Action 2: New label (the block has foreground pixels and is not connected to anything else)
				#define ACTION_2 img_labels_row[c] = lunique; \
									P[lunique] = lunique;        \
									lunique = lunique + 1;
				//Action 3: Assign label of block P
				#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2];
				// Action 4: Assign label of block Q
				#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c];
				// Action 5: Assign label of block R
				#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2];
				// Action 6: Assign label of block S
				#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2];
				// Action 7: Merge labels of block P and Q
				#define ACTION_7 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]);
				//Action 8: Merge labels of block P and R
				#define ACTION_8 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]);
				// Action 9 Merge labels of block P and S
				#define ACTION_9 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]);
				// Action 10 Merge labels of block Q and R
				#define ACTION_10 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]);
				// Action 11: Merge labels of block Q and S
				#define ACTION_11 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c], img_labels_row[c - 2]);
				// Action 12: Merge labels of block R and S
				#define ACTION_12 img_labels_row[c] = set_union(P, img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]);
				// Action 13: Merge labels of block P, Q and R
				#define ACTION_13 img_labels_row[c] = set_union(P, set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row_prev_prev[c + 2]);
				// Action 14: Merge labels of block P, Q and S
				#define ACTION_14 img_labels_row[c] = set_union(P, set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]);
				//Action 15: Merge labels of block P, R and S
				#define ACTION_15 img_labels_row[c] = set_union(P, set_union(P, img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
				//Action 16: labels of block Q, R and S
				#define ACTION_16 img_labels_row[c] = set_union(P, set_union(P, img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]);
			}
			// The following Directed Rooted Acyclic Graphs (DAGs) allow to choose which action to
			// perform, checking as few conditions as possible. Special DAGs are used for the first/last
			// line of the image and for single line images. Actions: the blocks label are provisionally
			// stored in the top left pixel of the block in the labels image.
			if (h == 1) {
				// Single line
				const PixelT * const img_row = img.ptr<PixelT>(0);
				LabelT * const img_labels_row = imgLabels.ptr<LabelT>(0);
				int c = -2;
				#include "./OpencvCode/ccl_bolelli_forest_singleline.inc.hpp"
			}
			else {
				// More than one line

				// First couple of lines
				{
					const PixelT * const img_row = img.ptr<PixelT>(0);
					const PixelT * const img_row_fol = (PixelT *)(((char*)img_row) + img.step.p[0]);
					LabelT * const img_labels_row = imgLabels.ptr<LabelT>(0);
					int c = -2;
					#include "./OpencvCode/ccl_bolelli_forest_firstline.inc.hpp"
				}

				// Every other line but the last one if image has an odd number of rows
				for (int r = 2; r < e_rows; r += 2) {
					// Get rows pointer
					const PixelT * const img_row = img.ptr<PixelT>(r);
					const PixelT * const img_row_prev = (PixelT *)(((char*)img_row) - img.step.p[0]);
					const PixelT * const img_row_prev_prev = (PixelT *)(((char*)img_row_prev) - img.step.p[0]);
					const PixelT * const img_row_fol = (PixelT *)(((char*)img_row) + img.step.p[0]);
					LabelT * const img_labels_row = imgLabels.ptr<LabelT>(r);
					LabelT * const img_labels_row_prev_prev = (LabelT *)(((char*)img_labels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);

					int c = -2;
					goto tree_0;

					#include "./OpencvCode/ccl_bolelli_forest.inc.hpp"
				}

				// Last line (in case the rows are odd)
				if (o_rows) {
					int r = h - 1;
					const PixelT * const img_row = img.ptr<PixelT>(r);
					const PixelT * const img_row_prev = (PixelT *)(((char*)img_row) - img.step.p[0]);
					const PixelT * const img_row_prev_prev = (PixelT *)(((char*)img_row_prev) - img.step.p[0]);
					LabelT * const img_labels_row = imgLabels.ptr<LabelT>(r);
					LabelT * const img_labels_row_prev_prev = (LabelT *)(((char*)img_labels_row) - imgLabels.step.p[0] - imgLabels.step.p[0]);
					int c = -2;
					#include "./OpencvCode/ccl_bolelli_forest_lastline.inc.hpp"
				}
			}

			// undef conditions and actions
			{
				#undef ACTION_1
				#undef ACTION_2
				#undef ACTION_3
				#undef ACTION_4
				#undef ACTION_5
				#undef ACTION_6
				#undef ACTION_7
				#undef ACTION_8
				#undef ACTION_9
				#undef ACTION_10
				#undef ACTION_11
				#undef ACTION_12
				#undef ACTION_13
				#undef ACTION_14
				#undef ACTION_15
				#undef ACTION_16

				#undef CONDITION_B
				#undef CONDITION_C
				#undef CONDITION_D
				#undef CONDITION_E

				#undef CONDITION_G
				#undef CONDITION_H
				#undef CONDITION_I
				#undef CONDITION_J
				#undef CONDITION_K

				#undef CONDITION_M
				#undef CONDITION_N
				#undef CONDITION_O
				#undef CONDITION_P

				#undef CONDITION_R
				#undef CONDITION_S
				#undef CONDITION_T
			}

			// Second scan + analysis
			LabelT nLabels = flattenL(P, lunique);
			sop.init(nLabels);

			int r = 0;
			for (; r < e_rows; r += 2) {
				// Get rows pointer
				const PixelT * const img_row = img.ptr<PixelT>(r);
				const PixelT * const img_row_fol = (PixelT *)(((char*)img_row) + img.step.p[0]);
				LabelT * const img_labels_row = imgLabels.ptr<LabelT>(r);
				LabelT * const img_labels_row_fol = (LabelT *)(((char*)img_labels_row) + imgLabels.step.p[0]);
				int c = 0;
				for (; c < e_cols; c += 2) {
					LabelT iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0) {
							img_labels_row[c] = iLabel;
							sop(r, c, iLabel);
						}
						else {
							img_labels_row[c] = 0;
							sop(r, c, 0);
						}
						if (img_row[c + 1] > 0) {
							img_labels_row[c + 1] = iLabel;
							sop(r, c + 1, iLabel);
						}
						else {
							img_labels_row[c + 1] = 0;
							sop(r, c + 1, 0);
						}
						if (img_row_fol[c] > 0) {
							img_labels_row_fol[c] = iLabel;
							sop(r + 1, c, iLabel);
						}
						else {
							img_labels_row_fol[c] = 0;
							sop(r + 1, c, 0);
						}
						if (img_row_fol[c + 1] > 0) {
							img_labels_row_fol[c + 1] = iLabel;
							sop(r + 1, c + 1, iLabel);
						}
						else {
							img_labels_row_fol[c + 1] = 0;
							sop(r + 1, c + 1, 0);
						}
					}
					else {
						img_labels_row[c] = 0;
						sop(r, c, 0);
						img_labels_row[c + 1] = 0;
						sop(r, c + 1, 0);
						img_labels_row_fol[c] = 0;
						sop(r + 1, c, 0);
						img_labels_row_fol[c + 1] = 0;
						sop(r + 1, c + 1, 0);
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					LabelT iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0) {
							img_labels_row[c] = iLabel;
							sop(r, c, iLabel);
						}
						else {
							img_labels_row[c] = 0;
							sop(r, c, 0);
						}
						if (img_row_fol[c] > 0) {
							img_labels_row_fol[c] = iLabel;
							sop(r + 1, c, iLabel);
						}
						else {
							img_labels_row_fol[c] = 0;
							sop(r + 1, c, 0);
						}
					}
					else {
						img_labels_row[c] = 0;
						sop(r, c, 0);
						img_labels_row_fol[c] = 0;
						sop(r + 1, c, 0);
					}
				}
			}
			// Last row if the number of rows is odd
			if (o_rows) {
				// Get rows pointer
				const PixelT * const img_row = img.ptr<PixelT>(r);
				LabelT * const img_labels_row = imgLabels.ptr<LabelT>(r);
				int c = 0;
				for (; c < e_cols; c += 2) {
					LabelT iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0) {
							img_labels_row[c] = iLabel;
							sop(r, c, iLabel);
						}
						else {
							img_labels_row[c] = 0;
							sop(r, c, 0);
						}
						if (img_row[c + 1] > 0) {
							img_labels_row[c + 1] = iLabel;
							sop(r, c + 1, iLabel);
						}
						else {
							img_labels_row[c + 1] = 0;
							sop(r, c + 1, 0);
						}
					}
					else {
						img_labels_row[c] = 0;
						sop(r, c, 0);
						img_labels_row[c + 1] = 0;
						sop(r, c + 1, 0);
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					LabelT iLabel = img_labels_row[c];
					if (iLabel > 0) {
						iLabel = P[iLabel];
						if (img_row[c] > 0) {
							img_labels_row[c] = iLabel;
							sop(r, c, iLabel);
						}
						else {
							img_labels_row[c] = 0;
							sop(r, c, 0);
						}
					}
					else {
						img_labels_row[c] = 0;
						sop(r, c, iLabel);
					}
				}
			}

			sop.finish();
			return nLabels;
		}//End function LabelingBolelli operator()
	};//End struct LabelingBolelli

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
		PZTRegions& operator= (PZTRegions&& t_other);
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
		* param0[i]: The index of connected domaim. It starts from 0.
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
		* brief    : Return the complement of a region.
		*/
		bool Complement();

		/*
		* brief    : Calculate the intersection of two regions. *** this->m_regionNum must be 1 ***
		* param0[i]: The other regions which will be intersected. *** t_regI::m_regionNum must be 1 ***
		* param1[o]: Result of the intersection.
		*/
		bool Intersection(const PZTRegions& t_regI, PZTRegions& t_regO) const;

		/*
		* brief    : Compute connected components of a region. *** m_regionNum must be 1 ***
		*/
		bool Connection();

		/*
		* brief    : Gain the union of all connected domains.
		*/
		bool Disconnection();

		/*
		* brief    : Fill up holes in one region. *** m_regionNum must be 1 有待商榷***  
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
		bool Closing(StructElement t_elm, unsigned int t_kernelWidth = 3, unsigned int t_kernelHeight = 3);

		void DisplayRegion(float t_factor = 1, bool t_isWrite = false, const std::string& t_name = "m_regions.bmp");

	private:
		bool _UpdataRegionNum();

		unsigned int _ConnectedComponents(cv::InputArray t_image, cv::OutputArray t_labels);

		/*
		* brief: 
		* 		_UpdataRegionFeaturesV1()	: It is only for blob.
		*    	_UpdataRegionFeaturesV2()	: It could be for various connected domains, but is time-consuming.
		*		_UpdataRegionFeaturesV3()	: Adopting multi-threaded technology to speed up.
		*		_UpdataRegionFeaturesV3_1() : 
		*/
		bool _UpdataRegionFeaturesV1(); 

		bool _UpdataRegionsFeaturesV2();
		RegionFeature _GainOneRegionFeaturesV2(cv::InputArray t_oneRegion);

		bool _UpdataRegionsFeaturesV3();
		RegionFeature _GainOneRegionFeaturesV3(uint32_t t_idx);

		bool __GainOneRegionFeatures(cv::InputArray t_oneRegion, const std::vector<cv::Point>& t_contour, RegionFeature& t_feature);
		bool _GainAreaFeature(cv::InputArray t_oneRegion, RegionFeature& t_feature);
		bool _GainContlengthFeature(const std::vector<cv::Point>& t_contour, RegionFeature& t_feature);
		bool _GainCircularityFeature(RegionFeature& t_feature);
		bool _GainRectangularityFeature(RegionFeature& t_feature);
		bool _GainMassCenterFeature(const std::vector<cv::Point>& t_contour, RegionFeature& t_feature);
		bool _GainBoundingRectangleFeature(const std::vector<cv::Point>& t_contour, RegionFeature& t_feature);
		bool _GainRotatedRectangleFeature(const std::vector<cv::Point>& t_contour, RegionFeature& t_feature);

/*
		bool _UpdataRegionsFeaturesV3(){
			// ...
			for(uint32_t idx = 1; idx <= m_regionNum; ++idx){
				//ThreadPoolObj.enqueue(&PZTRegions::_GainOneRegionFeaturesV3, this, idx);
			}

			// ThreadPoolObj.join
		}
		// 考虑 OpenCV 是否线程安全
		bool _GainOneRegionFeaturesV3(uint32_t t_idx){
			// such as
			cv::Mat oneRegion;
			cv::inRange(m_regions, t_idx, t_idx, oneRegion);
			cv::threshold(oneRegion, oneRegion, 0, 1, cv::THRESH_BINARY);

			_GainAreaFeature(oneRegion, (*m_featuresPtr)[t_idx]);
		}
*/
		static bool _HaveSameSize(const PZTRegions& t_reg1, const PZTRegions& t_reg2);

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

		//
		static ThreadPool								m_works;
	};

	bool TestCore();

	bool HalconDetection();

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// brief:
	//	Testing module
	//
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	class CoreTestor{
	public:
		static PZTRegions InitMemberComReg();
		static PZTImage InitMemberComImg();
		static bool TestFunc_UpdataRegionsFeaturesV2();
		static bool TestFunc_Connection();

	private:
		static PZTIMAGE::PZTImage						m_comImg;

		// ! PZTRegions object. Its Member m_regionNum is 1.
		static PZTIMAGE::PZTRegions						m_comReg;
	};
}

#endif
