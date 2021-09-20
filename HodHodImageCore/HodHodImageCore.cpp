// HodHodImageCore.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


void display(Mat& im, Mat& bbox)
{
	int n = bbox.rows;
	for (int i = 0; i < n; i++)
	{
		line(im, Point2i(bbox.at<float>(i, 0), bbox.at<float>(i, 1)), Point2i(bbox.at<float>((i + 1) % n, 0), bbox.at<float>((i + 1) % n, 1)), Scalar(255, 0, 0), 3);
	}
	imshow("Result", im);
}

void CalcHistogram(Mat& src)
{
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange[] = { range };
	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
	//calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	//normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		//line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
		//	Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
		//	Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("Source image", src);
	imshow("calcHist Demo", histImage);
}

void DetectQrRegions(Mat& im)
{
	Mat GrayImage;
	cvtColor(im, GrayImage, cv::COLOR_BGR2GRAY);
	Mat lower_red_hue_range;
	Mat upper_red_hue_range;
	
	//inRange(HSVImage, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
	//inRange(HSVImage, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);
	//inRange(HSVImage, Scalar(0, 120, 70), Scalar(10, 255, 255), lower_red_hue_range);
	//inRange(HSVImage, Scalar(170, 120, 70), Scalar(180, 255, 255), upper_red_hue_range);
	imshow("Gray", GrayImage);
	//imshow("HSV High", upper_red_hue_range);
	//imshow("HSV low", lower_red_hue_range);
	Mat mask;
	threshold(GrayImage, mask,0, 255, THRESH_BINARY );
	imshow("Mask", mask);
	int morph_size = 2;
	Mat element = getStructuringElement(MORPH_RECT, Size(3,3 ),Point(morph_size, morph_size));
	Mat Eroded;
	Mat Delteded;
	dilate(mask, Delteded, element, Point(-1, -1), 1);
	erode(Delteded, Eroded, element,Point(-1,-1), 1);
	Mat edges;
	Canny(Eroded, edges, 100, 255, 3, false);
	imshow("Canny edge detection", edges);
	// find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(edges, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>centers(contours.size());
	vector<float>radius(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		
		boundRect[i] = boundingRect(cv::Mat(contours[i]));
		
	}
	Mat drawing = Mat::zeros(im.size(), CV_8UC3);
	list<Mat> All;
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(0, 100, 255);
		if (boundRect[i].width > 180 && boundRect[i].height > 180)
		{
			//drawContours(drawing, contours_poly, (int)i, color);
			rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 1);
			
			if (i == 0)
			{
				All.push_back(im(boundRect[i]));
				imshow("ROI " + to_string(i), im(boundRect[i]));
				continue;
			}
			if (!(boundRect[i - 1].x == boundRect[i].x && boundRect[i - 1].y == boundRect[i].y && boundRect[i - 1].width == boundRect[i].width &&
				boundRect[i - 1].height == boundRect[i].height))
			{
				All.push_back(im(boundRect[i]));
				imshow("ROI " + to_string(i), im(boundRect[i]));
			}
			
		}
		
		//circle(drawing, centers[i], (int)radius[i], color, 2);
	}
	


}


int main()
{
	cout << "Hi!\n";
	cout << "HodHod Image Processor Starting :))";
	Mat inputImage;

	inputImage = imread("E:\\Test.png");
	DetectQrRegions(inputImage);
	QRCodeDetector qrDecoder = QRCodeDetector::QRCodeDetector();
	Mat bbox, rectifiedImage;
	string data = qrDecoder.detectAndDecode(inputImage, bbox, rectifiedImage);
	if (data.length() > 0)
	{
		cout << "Decoded Data : " << data << endl;
		display(inputImage, bbox);
		rectifiedImage.convertTo(rectifiedImage, CV_8UC3);
		imshow("Rectified QRCode", rectifiedImage);
		
	}
	waitKey(0);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
