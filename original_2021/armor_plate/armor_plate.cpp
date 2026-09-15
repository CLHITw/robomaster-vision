#include <opencv2/opencv.hpp>
#include "armor_plate.h"

using namespace cv;
using namespace std;

ArmorPlate::ArmorPlate()
{
	our_team_ = TEAMBLUE;
}
/*摄像头初始化*/
bool ArmorPlate::CamaraInit(int device)
{
	capture_armor_.open(device);
	if (!capture_armor_.isOpened())
	{
		cout << "视觉辅助摄像头打开失败" << endl;
		return false;
	}
	else
		return true;
}
/*图像预处理*/
void ArmorPlate::ImgPreprocess(const cv::Mat &src, cv::Mat &dst)//单独提取通道
{
	std::vector<Mat> img_channels;//通道
	split(src, img_channels);//通道分离

	if (our_team_)
	{
		Mat img_blue_channels;
		img_blue_channels = img_channels.at(0);
		imshow("blue_channels", img_blue_channels);
		img_blue_channels = img_blue_channels - img_channels.at(1) * 0.4 - img_channels.at(2) * 0.4;//去除白色影响
		imshow("blue", img_blue_channels);
	}
	else //我方为红队，对方蓝队
	{
		Mat img_blue_channels;
		img_blue_channels = img_channels.at(0);//蓝色通道
		//imshow("img_blue_channels", img_blue_channels);
		//img_red_channels = img_red_channels - img_channels.at(1)*0.4 - img_channels.at(2)*0.4;
		img_blue_channels = img_blue_channels - img_channels.at(2);
		//blur(img_blue_channels, img_blue_channels, Size(3, 3));

		//img_red_channels = img_red_channels*3;
		//imshow("blue_channels", img_blue_channels);
		double maxValue_gary;//最大灰度值
		minMaxLoc(img_blue_channels, 0, &maxValue_gary, 0, 0);//获取最大灰度值
		Mat imgBin;//定义二值化图像
		threshold(img_blue_channels, imgBin, maxValue_gary * 0.65, 255, THRESH_BINARY);
		//imshow("二值化", imgBin);
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(imgBin, dst, element, Point(-1, -1), 3);
		//erode(dst,dst,element,Point(-1,-1),4);
		//dilate(imgBin,dst,element,Point(-1,-1),3);
		imshow("RGB预处理", dst);
	}
}

void ArmorPlate:: FindArmor(cv::Mat &src, cv::Mat &dst, std::vector<cv::RotatedRect> &all, cv::RotatedRect &target)
{
	all.clear();//清空
	/*初始化*/
	target.center.x = 0;
	target.center.y = 0;
	target.size.width = 0;
	target.size.height = 0;
	target.angle = 0;//此时为空

	RotatedRect s, s_fitEllipse, s_minAreaRect;//用于筛选轮廓（最小矩形）
	vector <RotatedRect> ss;//筛选完存放的位置
	ss.clear();

	vector<vector<Point>> contours;//轮廓
	vector<Vec4i> hierarchy;//层次

	findContours(dst,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE);//寻找轮廓
	Mat drawing = Mat::zeros(dst.size(),CV_8UC3);
	RNG g_rng(12345);
	for (int i=0;i<contours.size();i++)
	{
		Scalar color = Scalar(g_rng.uniform(0,255),g_rng.uniform(0,255),g_rng.uniform(0,255));
		drawContours(drawing,contours,i,color,2,8,hierarchy,0,Point());//画出所有轮廓
		imshow("轮廓",drawing);
		if (contours[i].size() >= 10)//自己调
		{
			s_fitEllipse = fitEllipse(contours[i]);//椭圆拟合
			s_minAreaRect = minAreaRect(contours[i]);

			s.angle = s_fitEllipse.angle;//用椭圆的角度是不会产生歧义
			s.center = s_fitEllipse.center;
			if(s_minAreaRect.size.width > s_minAreaRect.size.height)
			{
				s.size.height = s_minAreaRect.size.width ;
				s.size.width = s_minAreaRect.size.height;
			}
			else
			{
				s.size.height = s_minAreaRect.size.height;
				s.size.width = s_minAreaRect.size.width;
			}//宽高扭转，方便理解
			
			if ((s.size.width / s.size.height ) > L_WH_RAT )
				continue;
			int x = s.center.x - s.size.width;
			if(x < 0)
				continue;
			int y = s.center.y - s.size.height;
			if(y < 0)
				continue;
			int w = s.size.width + s.size.width;
			if(w > dst.cols - x)
				continue;
			int h = s.size.height + s.size.height;
			if(w > dst.rows- y)
				continue;
			
			if((s.angle < 45 || s.angle > 135) && (s.size.height > 10) && (s.size.height < 150))//角度筛选
				ss.push_back(s);
			
		}
	}
	//判定是否为装甲
	vector<RotatedRect> armors;
	vector<ArmorBuilded> armor_SECOND;//装甲构成类
	ArmorBuilded armor_FIRST;//旋转矩形区域
	static float armor_center_x;
	static float armor_center_y;

	armors.clear();
	armor_SECOND.clear();
	int nL,nW;

	if (ss.size() < 2)//旋转矩形小于两个，直接返回，构成装甲需要两个灯条
	{
		target.center.x = 0;
		target.center.y = 0;
		target.size.width = 0;
		target.size.height = 0;
		target.angle = 0;
		all.push_back(target);
		armor_center_x = 0;
		armor_center_y = 0;
	}
	else
	{
		for(int i = 0; i < ss.size() - 1; i++)//ss是旋转矩形容器
		{
			for(int j =i + 1; j < ss.size(); j++)//两两比较
			{
				double height_diff = abs(ss[i].size.height - ss[j].size.height);//高度差
				double height_sum = ss[i].size.height + ss[j].size.height;//高度和
				double width_diff = abs(ss[i].size.width - ss[j].size.width);//宽度差
				double width_sum = ss[i].size.width + ss[j].size.width;//宽度和
				double angle_diff = fabs(ss[i].angle - ss[j].angle);//角度差
				double Y_diff = abs(ss[i].center.y - ss[j].center.y);//纵坐标差值
				double X_diff = abs(ss[i].center.x - ss[j].center.x);//横坐标差值
				double MH_diff = (min(ss[i].size.height,ss[j].size.height))*2/3;//高度差限幅
				double height_max = max(ss[i].size.height,ss[j].size.height);//最大高度

				if (Y_diff < MH_diff && X_diff < height_max*4 && (angle_diff < T_ANGLE_THRE || 180 - angle_diff <  T_ANGLE_THRE ) && height_diff / height_sum < T_HIGH_RAT && width_diff / width_sum <T_WHIDTH_RAT)//此处为满足筛选条件
				{
					armor_FIRST.armorS.center.x = ((ss[i].center.x + ss[j].center.x ) / 2);
					armor_FIRST.armorS.center.y = ((ss[i].center.y + ss[j].center.y ) / 2);
					armor_FIRST.armorS.angle = ((ss[i].angle + ss[j].angle )/2);
					if(180 - angle_diff < T_ANGLE_THRE180)//角度变换
						armor_FIRST.armorS.angle += 90;
					nL = (ss[i].size.height + ss[j].size.height) / 2;//高度求平均
					nW = sqrt((ss[i].center.x - ss[j].center.x) * (ss[i].center.x - ss[j].center.x) + (ss[i].center.y - ss[j].center.y) * (ss[i].center.y - ss[j].center.y));//宽度与旋转中心的联系
					if (nL < nW)//同上思路，长的为宽
					{
						armor_FIRST.armorS.size.height = nL;
						armor_FIRST.armorS.size.width = nW;
					}
					else
					{
						armor_FIRST.armorS.size.height = nW;
						armor_FIRST.armorS.size.width = nL;
					}
					if (Y_diff < nW / 3)
					{
						armor_FIRST.build1_No = i;
						armor_FIRST.build2_No = j;
						armor_FIRST.build_features[0] = angle_diff;
						armor_FIRST.build_features[1] =Y_diff;
						armor_FIRST.build_features[2] = height_diff;
						armor_FIRST.build_features[3] = width_diff;
						armor_SECOND.push_back(armor_FIRST);
					}
				}
				else if ((angle_diff < T_ANGLE_THREMIN || 180 - angle_diff < T_ANGLE_THRE180MIN) && Y_diff <MH_diff * 3 / 2 && X_diff < height_max * 4 && height_diff / height_sum < T_HIGH_RAT_ANGLE && width_diff /width_sum <T_WHIDTH_RAT_ANGLE)
				{
					armor_FIRST.armorS.center.x = ((ss[i].center.x + ss[j].center.x ) / 2);
					armor_FIRST.armorS.center.y = ((ss[i].center.y + ss[j].center.y ) / 2);
					armor_FIRST.armorS.angle = ((ss[i].angle + ss[j].angle )/2);
					if(180 - angle_diff < T_ANGLE_THRE180)
						armor_FIRST.armorS.angle += 90;
					nL = (ss[i].size.height + ss[j].size.height) / 2;
					nW = sqrt((ss[i].center.x - ss[j].center.x) * (ss[i].center.x - ss[j].center.x) + (ss[i].center.y - ss[j].center.y) * (ss[i].center.y - ss[j].center.y));
					if (nL < nW)
					{
						armor_FIRST.armorS.size.height = nL;
						armor_FIRST.armorS.size.width = nW;
					}
					else
					{
						armor_FIRST.armorS.size.height = nW;
						armor_FIRST.armorS.size.width = nL;
					}
					if (Y_diff < nW / 2)
					{
						armor_FIRST.build1_No = i;
						armor_FIRST.build2_No = j;
						armor_FIRST.build_features[0] = angle_diff;
						armor_FIRST.build_features[1] =Y_diff;
						armor_FIRST.build_features[2] = height_diff;
						armor_FIRST.build_features[3] = width_diff;
						armor_SECOND.push_back(armor_FIRST);
					}
				}
				else if ((angle_diff < 3 || 180 - angle_diff < 2) && Y_diff < MH_diff * 2 && X_diff < height_max * 4)
				{
					armor_FIRST.armorS.center.x = ((ss[i].center.x + ss[j].center.x ) / 2);
					armor_FIRST.armorS.center.y = ((ss[i].center.y + ss[j].center.y ) / 2);
					armor_FIRST.armorS.angle = (ss[i].angle + ss[j].angle )/2;
					if(180 - angle_diff < T_ANGLE_THRE180)
						armor_FIRST.armorS.angle += 90;
					nL = (ss[i].size.height + ss[j].size.height) / 2;
					nW = sqrt((ss[i].center.x - ss[j].center.x) * (ss[i].center.x - ss[j].center.x) + (ss[i].center.y - ss[j].center.y) * (ss[i].center.y - ss[j].center.y));
					if (nL < nW)
					{
						armor_FIRST.armorS.size.height = nL;
						armor_FIRST.armorS.size.width = nW;
					}
					else
					{
						armor_FIRST.armorS.size.height = nW;
						armor_FIRST.armorS.size.width = nL;
					}
					if ((abs(ss[i].center.y - ss[j].center.y) < nW / 2))
					{
						armor_FIRST.build1_No = i;
						armor_FIRST.build2_No = j;
						armor_FIRST.build_features[0] = angle_diff;
						armor_FIRST.build_features[1] =Y_diff;
						armor_FIRST.build_features[2] = height_diff;
						armor_FIRST.build_features[3] = width_diff;
						armor_SECOND.push_back(armor_FIRST);
					}
				}
				else if ((angle_diff < 3 || 180 - angle_diff < 2) && Y_diff < MH_diff *3 && X_diff <height_max * 5)
				{
					armor_FIRST.armorS.center.x = ((ss[i].center.x + ss[j].center.x ) / 2);
					armor_FIRST.armorS.center.y = ((ss[i].center.y + ss[j].center.y ) / 2);
					armor_FIRST.armorS.angle = (ss[i].angle + ss[j].angle )/2;
					if(180 - angle_diff < T_ANGLE_THRE180)
						armor_FIRST.armorS.angle += 90;
					nL = (ss[i].size.height + ss[j].size.height) / 2;
					nW = sqrt((ss[i].center.x - ss[j].center.x) * (ss[i].center.x - ss[j].center.x) + (ss[i].center.y - ss[j].center.y) * (ss[i].center.y - ss[j].center.y));
					if (nL < nW)
					{
						armor_FIRST.armorS.size.height = nL;
						armor_FIRST.armorS.size.width = nW;
					}
					else
					{
						armor_FIRST.armorS.size.height = nW;
						armor_FIRST.armorS.size.width = nL;
					}
					if (Y_diff < nW / 2)
					{
						armor_FIRST.build1_No = i;
						armor_FIRST.build2_No = j;
						armor_FIRST.build_features[0] = angle_diff;
						armor_FIRST.build_features[1] = Y_diff;
						armor_FIRST.build_features[2] = height_diff;
						armor_FIRST.build_features[3] = width_diff;
						armor_SECOND.push_back(armor_FIRST);//将找出的装甲的旋转矩形保存到vector
					}
				}
			}	
		}
		if (armor_SECOND.size() < 1)
		{
			/*int ss_width = 0;
			int ss_ID = 0;
			for(unsigned int SSS = 0; SSS < ss.size();SSS++)
			{
				if(ss[SSS].size.width > ss_width && (ss[SSS].size.width / ss[SSS].size.height) < 0.4 && )
			}*/
			cout<<"no armors!"<<endl;
		}
		else if (armor_SECOND.size() == 1)
		{
			target = armor_SECOND[0].armorS;
			all.push_back(armor_SECOND[0].armorS);
			armor_center_x = target.center.x;
			armor_center_y = target.center.y;
		}
		else
		{
			cout<<"too many armors!"<<endl;
		}
	}
}

void DrawAll(vector<RotatedRect> rect, Mat img)
{
	for(int i = 0; i < rect.size(); i++)
	{
		Point2f pp[4];
		rect[i].points(pp);//计算二维盒子顶点
		line(img,pp[0],pp[1],CV_RGB(255,255,255),1,8,0);
		line(img,pp[1],pp[2],CV_RGB(255,255,255),1,8,0);
		line(img,pp[2],pp[3],CV_RGB(255,255,255),1,8,0);
		line(img,pp[3],pp[0],CV_RGB(255,255,255),1,8,0);
	}
}

void DrawTarget(RotatedRect box, Mat img)
{
	Point2f pts[8];
	pts[0].x = box.center.x;
	pts[0].y = box.center.y - 10;
	pts[1].x = box.center.x;
	pts[1].y = box.center.y + 10;
	pts[2].x = box.center.x - 10;
	pts[2].y = box.center.y ;
	pts[3].x = box.center.x + 10;
	pts[3].y = box.center.y;

	pts[4].x = img_center_x;
	pts[4].y = img_center_y - 10;
	pts[5].x = img_center_x;
	pts[5].y = img_center_y + 10;
	pts[6].x = img_center_x - 10;
	pts[6].y = img_center_y;
	pts[7].x = img_center_x + 10;
	pts[7].y = img_center_y;
	line(img,pts[0],pts[1],CV_RGB(0,255,0),2,8,0);
	line(img,pts[2],pts[3],CV_RGB(0,255,0),2,8,0);
	line(img,pts[4],pts[5],CV_RGB(255,255,255),2,8,0); 
	line(img,pts[6],pts[7],CV_RGB(255,255,255),2,8,0);
}
float BulletModel(float x,float v,float angle)
{
	float t,y;
	int init_k_ = 1;
	t = (float)((exp(init_k_ * x) - 1) / (init_k_ * v * cos(angle)));
	y = (float)(v * sin(angle) * t - 9.8 *t * t / 2);
	return y;
}
float GetPitch(float x,float y,float v)
{
	float y_temp,y_actual,dy;
	float a;
	y_temp = y;
	for(int i = 0;i < 20;i++)
	{
		a = (float) atan2(y_temp,x);
		y_actual = BulletModel(x,v,a);
		dy = y - y_actual;
		y_temp = y_temp + dy;
		if(fabsf(dy) < 0.001)
		{
			break;
		}
	}
	return a;
}
void display (Mat img,RotatedRect rect)
{
	double camD[9] = {2.51913284939712e+03, 0.749381889519802,614.938139894739, 0.,
           2.51717058777089e+03, 513.456628760370, 0., 0., 1. };
	Mat camera_matrix = Mat(3,3,CV_64FC1,camD);

  	double distCoeffD[5]={-0.0401129361656092, -0.506814027965233, 0.00106468623922559, -0.000269864370417353, 10.1144125951595};
	Mat distortion_coefficients = Mat(5,1.CV_64FC1,distCoeffD);

	vector<Point2f> Points2D;
	Points2D.push_back(Point2f(img_center_x,img_center_y));
	Points2D.push_back(Point2f(img_center_x,img_center_y+10));
	Points2D.push_back(Point2f(rect.center.x,rect.center.y));
	Points2D.push_back(Point2f(rect.center.x,rect.center.y+10));

	vector<Point3f> Points3D;
	Points3D.push_back(Point3f(-67,-27,0));
	Points3D.push_back(Point3f(-67,27,0));
	Points3D.push_back(Point3f(67,-27,0));
	Points3D.push_back(Point3f(67,27,0));

	Mat rvec = Mat::zeros(3,1,CV_32FC1);

	Mat tvec = Mat::zeros(3,1,CV_32FC1);

	solvePnP(Points3D,Points2D,camera_matrix,distortion_coefficients,rvec,tvec,false,SOLVEPNP_P3P);
	Mat rotMat;
	Rodrigues(rvec,rotMat);
	Eigen::Matrix3f R_n;
	Eigen::Vector3f T_n;
	cv2eigen(rotMat,R_n);
	cv2eigen(tvec,T_n);
	Eigen::Vector3f P_oc;
	P_oc=-R_n.inverse()*T_n;
	char a[20];
	char b[20];
	char c[20];
	sprintf(a,"%f",x);
	sprintf(b,"%f",y);
	sprintf(c,"%f",z);
	putText(img,a,Point(50, 100),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255));
	putText(img,b,Point(50, 150),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255));
	putText(img,c,Point(50, 200),FONT_HERSHEY_COMPLEX,1,Scalar(255,255,255));
	float pitch,yaw;
	pitch = -GetPitch((z + z) / 100, -(y + y) / 100 ,26);
	yaw = -(float) (atan2(x + x , z + z));
	char temp1[100];
	sprintf(temp1,"pitch:%f,yaw:%f",pitch,yaw);
	Point pre_pt1;
	pre_pt1 = Point(20,20);
	putText(image,temp1,pre_pt1,FONT_HERSHEY_SIMPLE,1,Scalar(255,255,255));

}


/*辅助射击*/
void ArmorPlate::AutoShoot()
{
	//armor_image_=imread("1.png");

	ImgPreprocess(armor_image_, pre_image_);
	FindArmor(armor_image_,pre_image_,all_target_,target_);

	DrawAll(all_target_, armor_image_);
	DrawTarget(	target_,armor_image_);
	display(armor_image_,all_target_);
	imshow("原图", armor_image_);
	//imshow(预处理图",pre_image_);
	waitKey(1);
}
