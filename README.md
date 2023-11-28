# MiniLEDPanel Detection

## 组成模块
- Core              ：核心图像数据类型(Image/Region/XDL)模块。
- ImgProc           ：图像处理算子模块。
- DefectDetection   ：面板检测算法模块。
- Test              ：测试模块。

## 项目进程



## Opencv 库注意点
### cv::Mat
* 当 cv::Mat 对象被赋值/拷贝构造后，其 cv::Mat::data 与初始化它的对象的相同。
* 成员变量 cv::Mat::u::refcount 记录成员变量 cv::Mat::data 被引用多少次。当 cv::Mat 对象被赋值/拷贝构造后 refcount 加 1，当被析构后 refcount 减 1 (类似于智能指针引用计数)。

### Color Space
* cv::cvtColor() 只接受 1 通道的 Bayer 图像。

### 关于边界 border 与曲线 curve 的感悟
>>> #### 常见的 API

做的是什么？现在做的怎么样？后续 难点？
