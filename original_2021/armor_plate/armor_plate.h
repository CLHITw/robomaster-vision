#pragma once
#ifndef ARMOR_PLATE_H_INCLUDED
#define ARMOR_PLATE_H_INCLUDED

#include<opencv2/opencv.hpp>
#include<iostream>

#define img_center_x 640
#define img_center_y 240//

#define T_ANGLE_THRE 5//灯柱正向最大角度差
#define T_ANGLE_THRE180 3//灯条反向最大角度差
#define T_ANGLE_THREMIN 3//灯条正向最小角度差
#define T_ANGLE_THRE180MIN 2//灯条反向最小角度差

#define T_HIGH_RAT 0.2//灯带最大高差比 灯柱高度差是最大灯柱的1/3
#define T_HIGH_RAT_ANGLE 0.34//灯带符合最小角度高差比 灯柱高度差是最大灯柱的1/2

#define T_WHIDTH_RAT 0.4//灯带最大宽差比 灯柱宽度差是最大灯柱的3/5
#define T_WHIDTH_RAT_ANGLE 0.55//灯带最大宽差比

#define L_WH_RAT 0.8//灯条宽高比

#define TEAMBLUE 0
constexpr auto TEAMRED = 1;


class ArmorPlate
{
public:
	cv::Mat armor_image_;//视觉辅助原图
	cv::Mat pre_image_;//预处理图像
	cv::VideoCapture capture_armor_;//视觉辅助摄像头
	
	int our_team_;//红或蓝

	ArmorPlate();//构造函数
	bool CamaraInit(int device);//辅助摄像头初始化
	void AutoShoot();//外部调用函数
	
private:
	
	std::vector<cv::RotatedRect> all_target_;
	cv::RotatedRect target_;
	void ImgPreprocess(const cv::Mat& src, cv::Mat& dst);//图像预处理
	void FindArmor(cv::Mat& src, cv::Mat& dst, std::vector <cv::RotatedRect>& all,cv::RotatedRect& target);//定位装甲板
};

class ArmorBuilded//装甲构成
{
public:
	cv::RotatedRect armorS;
	int build1_No = 0;
	int build2_No = 0;
	int build_features[4];//角度差，高度坐标差，高度差，宽度差
	int vot = 0;
	
};
#endif
