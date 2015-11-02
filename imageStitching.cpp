#include <stdio.h>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

vector <DMatch> matcher(Mat img1_descriptor, Mat img2_descriptor);
Mat getH(vector <DMatch> matchPoints, vector <KeyPoint> img1_keypoint, vector <KeyPoint> img2_keypoint);

int main(int argc, char** argv)
{
	const int num = argc - 1;
	Mat *img = new Mat[num];
	Mat *gray_img = new Mat[num];
	int **imgMatch = new int*[num];					// imgMatch : 두 이미지간의 match 여부
	int *imgUse = new int[num];

	int *imgStack = new int[num];
	int rear = 0;

	//변수 초기화

	for (int i = 0; i< num; i++){
		imgMatch[i] = new int[num];
		imgUse[i] = 0;								//0: not handled, 1: handled
	}

	for (int i = 0; i < num; i++){
		for (int j = 0; j < num; j++){
			imgMatch[i][j] = 0;						// 0: no match , 1: match
		}
	}

	for (int i = 0; i < num; i++){
		//Load the images
		img[i] = imread(argv[i + 1], IMREAD_COLOR);

		if (!img[i].data){
			cout << "Could not open or find the images" << endl;
			return -1;
		}
		//Convert to Grayscale
		cvtColor(img[i], gray_img[i], COLOR_BGR2GRAY);
	}

	int minHessian = 400;
	vector <KeyPoint> *keypoint = new vector <KeyPoint>[num];
	Mat *descriptor = new Mat[num];
	//Mat *dest= new Mat[num];
	SiftFeatureDetector detector(minHessian);
	SiftDescriptorExtractor extractor;

	for (int i = 0; i < num; i++){
		detector.detect(gray_img[i], keypoint[i]);
		extractor.compute(gray_img[i], keypoint[i], descriptor[i]);
	}

	vector <DMatch> matchPoints[10][10];
	int max = 0;
	int maxi, maxj;
	for (int i = 0; i < num; i++){
		for (int j = i + 1; j < num; j++){
			matchPoints[i][j] = matcher(descriptor[i], descriptor[j]);
			matchPoints[j][i] = matcher(descriptor[j], descriptor[i]);

			if (max < matchPoints[i][j].size()){
				max = matchPoints[i][j].size();
				maxi = i;
				maxj = j;
			}

			if (matchPoints[i][j].size() >= 4){
				imgMatch[i][j] = matchPoints[i][j].size();
				imgMatch[j][i] = matchPoints[i][j].size();
			}
		}
	}
	//
	int rows = 0, cols = 0;

	for (int i = 0; i < num; i++){
		rows += img[i].rows;
		cols += img[i].cols;
	}
	// * ([1,0,(cols/2 - img[0].cols/2)][0,1,(rows/2 - img[0].rows/2)][0,0,1]);
	Mat *H = new Mat[num];
	//질문: 이것처럼 matchPoints vector의 크기가 가장 큰 것을 고르는 것이 좋을까 아니면 imgMatch가 큰 것을 고르는 것이 좋을까? 아니면 imgMatch 와 metchPoint둘다 고려?
	H[maxi] = Mat::eye(3, 3, CV_32F);
	H[maxi].at<float>(0, 2) = img[0].cols / num;
	H[maxi].at<float>(1, 2) = img[0].rows / num;
	H[maxi].convertTo(H[maxi], 6, 1, 0);

	imgUse[maxi] = 1;

	H[maxj] = H[maxi] * getH(matchPoints[maxj][maxi], keypoint[maxj], keypoint[maxi]);
	imgUse[maxj] = 1;

	imgStack[rear++] = maxi;
	imgStack[rear++] = maxj;

	int p, q;
	max = 1;

	while (max){
		/*
		*1단계 : imgUse값이 0이고 imgUse값이 1인 그림들과 연결되어있는(imgMatch탐색) 그림을 찾는다. 그 그림을 p라고 두자.
		*2단계 : p와 imgUse값이 1인 그림중에서 matchPoint[p][?].size() 값이 최대인 그림 q를 고른다.
		*3단계 : q의 호모그래피를 사용하여 p의 호모그래피값을 고쳐준다.
		*4단계 : imgUse of p = 0; 그리고 3단계로 돌아가서 반복한다. 모든 imgUse값들이 1이 될때까지!
		*/

		max = 0;

		for (int i = 0; i < num; i++){
			if (imgUse[i] == 1){
				for (int j = 0; j < num; j++){
					if (imgUse[j] == 0 && imgMatch[i][j] > max){
						p = i;
						q = j;							//newly warping image
						max = imgMatch[i][j];
					}
				}
			}
		}

		if (max == 0)
			break;

		H[q] = H[p] * getH(matchPoints[q][p], keypoint[q], keypoint[p]);
		imgUse[q] = 1;
		imgStack[rear++] = q;
	}

	Mat tmp1(cols, rows, CV_32F);
	Mat tmp2(cols, rows, CV_32F);
	Mat result(cols, rows, CV_32F);

	for (int i = 0; i < rear; i++){
		if (i == 0){
			warpPerspective(img[imgStack[i]], result, H[imgStack[i]], Size(cols, rows));
			//imshow("Result", result);
			//waitKey(0);
		}
		else{
			tmp1 = result;
			warpPerspective(img[imgStack[i]], tmp2, H[imgStack[i]], Size(cols, rows));
			addWeighted(tmp1, 0.5, tmp2, 0.7, 0.0, result);
			//imshow("Result", result);
			//waitKey(0);
		}
	}

	imshow("Result", result);
	waitKey(0);

	return 0;

}

vector <DMatch> matcher(Mat img1_descriptor, Mat img2_descriptor){

	Mat distance(img1_descriptor.rows, img2_descriptor.rows, CV_32F); //0.f

	float dist;
	int min = 10000;

	for (int i = 0; i < img1_descriptor.rows; i++){
		for (int j = 0; j < img2_descriptor.rows; j++){

			dist = 0;

			for (int k = 0; k < img1_descriptor.cols; k++)
				dist += abs(img1_descriptor.at<float>(i, k) - img2_descriptor.at<float>(j, k));

			distance.row(i).col(j) = dist;

			if (dist < min) min = dist;
		}
	}

	vector<DMatch> matchPoints;
	DMatch temp;

	for (int i = 0; i < distance.rows; i++){
		temp.distance = 10000;

		for (int j = 0; j < distance.cols; j++){
			if (distance.at<float>(i, j) < min * 2 && distance.at<float>(i, j) < temp.distance){
				temp.distance = distance.at<float>(i, j);
				temp.queryIdx = i;
				temp.trainIdx = j;
				temp.imgIdx = 0;
			}
		}

		if (temp.distance < 10000){
			for (int k = 0; k < distance.rows; k++){
				distance.row(k).col(temp.trainIdx) = 10000;
			}
			matchPoints.push_back(temp);
		}
	}

	return matchPoints;
}

Mat getH(vector <DMatch> matchPoints, vector <KeyPoint> img1_keypoint, vector <KeyPoint> img2_keypoint){

	vector< Point2f > img1_point;
	vector< Point2f > img2_point;

	for (int i = 0; i < matchPoints.size(); i++){
		//-- Get the keypoints from the good matches
		img1_point.push_back(img1_keypoint[matchPoints[i].queryIdx].pt);
		img2_point.push_back(img2_keypoint[matchPoints[i].trainIdx].pt);
	}

	// Find the Homography Matrix
	Mat H = findHomography(img1_point, img2_point, CV_RANSAC);

	return H;
}
