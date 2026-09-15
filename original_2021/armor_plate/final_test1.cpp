#include<opencv2/opencv.hpp>
#include"armor_plate.h"

using namespace cv;
using namespace std;

#define ARMOR_DEVICE 0

bool CamerRead(ArmorPlate& armor_param);

int main ()
{
	ArmorPlate armor;
	//Mat img;
	//armor.CamaraInit(ARMOR_DEVICE);
   	VideoCapture capture("/home/yhyzshrby/video1.avi");
	armor.capture_armor_ = capture;
	while(1)
	{
		if(!CamerRead(armor))
		{
			continue;
		}
		armor.AutoShoot();
		
	}
	//waitKey(0);
	//system("pause");
	return 0;
}


bool CamerRead(ArmorPlate& armor_param)
{
	armor_param.capture_armor_.read(armor_param.armor_image_);
	if(!armor_param.armor_image_.data)
	{
		cout<<"视觉辅助摄像头没有读取到头像!"<<endl;
		armor_param.CamaraInit(ARMOR_DEVICE);//视觉辅助摄像头初始化（防止崩溃）	
		return false;
	}
	else
		return true;//要有返回值给上面调用
}
