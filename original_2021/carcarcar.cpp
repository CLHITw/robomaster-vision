
    vector<int>labels;            
    double eps=1;
    std::vector<int>* weights;
    int nclasses = partition(rectList, labels, SimilarRects(eps));
   
    std::vector<double>* levelWeights;
    std::vector<Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    std::vector<int> rejectLevels(nclasses, 0);
    std::vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
 
    bool useDefaultWeights = false;
 
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }
    else
        useDefaultWeights = true;
    // 计算每一类别的平均矩形框位置，即每一个类别最终对应一个矩形框
    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }
 
    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();
    // 再次过滤上面分类中得到的所有矩形框
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        int l1 = rejectLevels[i];
 
        // filter out rectangles which don't have enough similar rectangles
        // 将每一类别中矩形框个数较少的类别过滤掉。
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        // 将嵌在大矩形框内部的小矩形框过滤掉。最后剩下的矩形框为聚类的结果。
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];
 
            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];
 
            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );
 
            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }
 
        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(useDefaultWeights ? n1 : l1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
        
        
        
        
        
        
        
        
        
        
        
     #include<opencv2/opencv.hpp>
#include <fstream>
#include<iostream>
 
using namespace cv;
 
using namespace std;
 

 
int main()
{
    VideoCapture capture("/home/wangqz/video1.avi");
    Mat frame;
if  (capture.isOpened())  //判断视频是否成功打开
{
     //capture.grab() 从视频文件或捕获设备中抓取下一个帧
 
    while (capture.grab())
    {
 
    capture  >> frame;Mat gray;
    cvtColor(frame,gray,COLOR_BGR2GRAY);
    blur(gray,gray,Size(3,3));
    GaussianBlur(gray,gray,Size(9,9),2,2);
    medianBlur(gray,gray,15); 
    Mat thout;
    vector<vector<Point>>contours;
    vector<Vec4i>hierarachy;
    threshold(gray,thout,40,255,THRESH_BINARY);

    findContours(thout,contours,hierarachy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point(0,0));
    vector<vector<Point>>contours_poly(contours.size());
    vector<Rect>boundRect(contours.size());
    vector<Point> labels;
    vector<int>hot;
                for(unsigned int i=0;i<contours.size();i++)
                {
                        approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
                        boundRect[i]=boundingRect(Mat(contours_poly[i]));
                }
    Mat draw=Mat:: zeros(thout.size(),CV_8UC3);
               for(int unsigned i=0;i<contours.size();i++)
               {
                     Scalar color=Scalar(255);
  
                     rectangle(frame,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
                    
               }
            
                imshow("car1",frame);
                waitKey(50);
    } 
}
return 0;
 }
 
 
 
 
 
 
 内参: [2.51913284939712e+03, 0.749381889519802,614.938139894739, 0.,
           2.51717058777089e+03, 513.456628760370, 0., 0., 1. ]

畸变：data: [-0.0401129361656092, -0.506814027965233, 0.00106468623922559, -0.000269864370417353, 10.1144125951595]
  
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 //函数主要功能为对矩形框进行融合操作。
void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps,
                     std::vector<int>* weights, std::vector<double>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }
 
    std::vector<int> labels;
	// 调用partition函数，将所有的矩形框初步分为几类，其中labels为每个矩形框对应的类别编号，eps为判断两个矩形框是否属于
	// 同一类的控制参数。如果两个矩形框的四个相应顶点的差值的绝对值都在deta范围内，则认为属于同一类，否则是不同类。
    int nclasses = partition(rectList, labels, SimilarRects(eps));
 
    std::vector<Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    std::vector<int> rejectLevels(nclasses, 0);
    std::vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }
 
    bool useDefaultWeights = false;
 
    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }
    else
        useDefaultWeights = true;
	// 计算每一类别的平均矩形框位置，即每一个类别最终对应一个矩形框
    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }
 
    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();
	// 再次过滤上面分类中得到的所有矩形框
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        int l1 = rejectLevels[i];
 
        // filter out rectangles which don't have enough similar rectangles
        // 将每一类别中矩形框个数较少的类别过滤掉。
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        // 将嵌在大矩形框内部的小矩形框过滤掉。最后剩下的矩形框为聚类的结果。
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];
 
            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];
 
            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );
 
            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }
 
        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(useDefaultWeights ? n1 : l1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}
  
