# PCVch04

记录学习Python Computer Vision的过程

第五次

## 增强现实

### 前言

随着互联网的不断发展，各种需求不断的被发现和改善，Yahoo、Google、Youtube，Facebook、Twitter等的出现掀起一次又一次的热潮。那么下一个热潮将会是什么呢？
个人认为，下一个热潮很有可能将会是，在这里想给大家介绍的，最近在世界各地备受关注的**增强现实**(Augmented Reality)技术。
首先**增强现实**是什么呢？
关于**增强现实**有很多种定义方法，我自己的理解是：**增强现实**是在人们接触到的真实世界上，叠加虚拟电子信息，对真实世界的信息进行增强或者扩张，帮助利用者们从事各种活动。

**增强现实**的适用范围比较广，目前在医疗，军事和游戏等领域已经有被使用的实例：
1、互联网： 将现有互联网信息附加显示在现实信息中
2、游戏娱乐：结合于现实的游戏，让玩家感受更真实
3、医疗：辅助精确的定位手术部位
4、军事：通过方位识别，获取所在地的相关地理信息等
5、古迹复原：在文化古迹的遗址上进行虚拟原貌恢复
6、工业：工业设备的相关信息显示，比如宽度，属性等
7、电视：在电视画面上显示辅助信息
8、旅游：对正在观看的风景上显示说明信息
9、建设：将建设规划效果叠加在真实场景，更加直观

### 准备工作

在进行增强现实的实验之前，我们必须先安装两个工具包：**PyGame**、**OpenGL**。

#### PyGame的安装

Pygame是跨平台Python模块，专为电子游戏设计，包含图像、声音。建立在SDL基础上，允许实时电子游戏研发而无需被低级语言（如[机器语言](https://baike.baidu.com/item/%E6%9C%BA%E5%99%A8%E8%AF%AD%E8%A8%80/2019225)和[汇编语言](https://baike.baidu.com/item/%E6%B1%87%E7%BC%96%E8%AF%AD%E8%A8%80/61826)）束缚。

SDL是用C写的，不过它也可以使用C++进行开发，当然还有很多其它的语言，Pygame就是Python中使用它的一个库。Pygame已经存在很多时间了，许多优秀的程序员加入其中，把Pygame做得越来越好。

**如果你的电脑安装的是python（x，y）版本的话，你都可以省去安装pygame和OpenGL的步骤**。

那么我们以没安装python（x，y）的情况来简单说下安装的过程

1.  首先下载[pygame](<https://www.lfd.uci.edu/~gohlke/pythonlibs/#pygame>)。如果你的python是2.7版本而且电脑是64位的话，那么你就选择<u>pygame-1.9.4-cp27-cp27m-win_amd64.whl</u>，同理你如果是其他版本的python就选择相应的whl就行了，然后点击，即可下载。

2. 下载下来之后在cmd命令框中cd进入 pygame-1.9.4-cp27-cp27m-win_amd64.whl的安装路径，输入 pip install pygame-1.9.4-cp27-cp27m-win_amd64.whl即可。

3. 对安装的pygame进行测试，在cmd中输入python，然后输入import pygame，如果没有报错即表示安装成功。

   ![](https://github.com/zengqq1997/PCVch04/blob/master/py1.jpg)

#### OpenGL的安装

**OpenGL**（英语：*Open Graphics Library*，译名：**开放图形库**或者“开放式图形库”）是用于[渲染](https://baike.baidu.com/item/%E6%B8%B2%E6%9F%93)[2D](https://baike.baidu.com/item/2D)、[3D](https://baike.baidu.com/item/3D)[矢量图形](https://baike.baidu.com/item/%E7%9F%A2%E9%87%8F%E5%9B%BE%E5%BD%A2)的跨[语言](https://baike.baidu.com/item/%E8%AF%AD%E8%A8%80)、[跨平台](https://baike.baidu.com/item/%E8%B7%A8%E5%B9%B3%E5%8F%B0)的[应用程序编程接口](https://baike.baidu.com/item/%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F%E7%BC%96%E7%A8%8B%E6%8E%A5%E5%8F%A3)（API）。这个接口由近350个不同的函数调用组成，用来从简单的图形比特绘制复杂的三维景象。而另一种程序接口系统是仅用于[Microsoft Windows](https://baike.baidu.com/item/Microsoft%20Windows)上的[Direct3D](https://baike.baidu.com/item/Direct3D)。OpenGL常用于[CAD](https://baike.baidu.com/item/CAD)、[虚拟实境](https://baike.baidu.com/item/%E8%99%9A%E6%8B%9F%E5%AE%9E%E5%A2%83)、科学可视化程序和电子游戏开发。

OpenGL的高效实现（利用了图形加速硬件）存在于[Windows](https://baike.baidu.com/item/Windows)，部分[UNIX](https://baike.baidu.com/item/UNIX)平台和[Mac OS](https://baike.baidu.com/item/Mac%20OS)。这些实现一般由显示设备厂商提供，而且非常依赖于该厂商提供的硬件。[开放源代码](https://baike.baidu.com/item/%E5%BC%80%E6%94%BE%E6%BA%90%E4%BB%A3%E7%A0%81)库[Mesa](https://baike.baidu.com/item/Mesa)是一个纯基于软件的图形API，它的代码兼容于OpenGL。但是，由于许可证的原因，它只声称是一个“非常相似”的API。

同理，如果你安装的是python（x，y）时，可忽略此步骤。

简单安装步骤

1. 首先下载[OpenGL](<https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl>)
2. 如果想当然地使用 pip 如下所示安装，可能会有一些麻烦。

```python
pip install pyopengl
```

3. 当我这样安装之后，运行 OpenGL 代码，得到了这样的错误信息

```python
NullFunctionError: Attempt to call an undefined function glutInit, check for bool(glutInit) before calling
```

4. 原来，pip 默认安装的是32位版本的pyopengl，而我的操作系统是64位的。建议点击[这里](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl)下载适合自己的版本，直接安装.whl文件。我是这样安装的：

```python
pip install PyOpenGL-3.1.3b2-cp27-cp27m-win_amd64.whl
```

5. 对安装的OpenGL进行测试，在cmd中输入python，然后输入import OpenGL，如果没有报错即表示安装成功。

![](https://github.com/zengqq1997/PCVch04/blob/master/gl1.jpg)

### 开始实例

首先用如下代码来提取图像的SIFT特征，然后用RANSAC算法稳健地估计单应性矩阵

```python
sift.process_image(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\code\bookcode\pcv-book-code-master\pcv-book-code-master\ch04\ar_cup\chapter4\book_frontal.JPG', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')

sift.process_image(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\code\bookcode\pcv-book-code-master\pcv-book-code-master\ch04\ar_cup\chapter4\book_perspective.JPG', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

# 匹配特征，并计算单应性矩阵
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)
```

那么接下来就是增强现实的实现，首先我们需要计算照相机的标定矩阵，即获得标定矩阵K

实现代码如下

```python
def my_calibration(sz):
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K
```

接下来将照相机矩阵转换为OpenGL中的投影矩阵

```python
def set_projection_from_camera(K): 
	glMatrixMode(GL_PROJECTION) 
	glLoadIdentity()
	fx = K[0,0] 
	fy = K[1,1] 
	fovy = 2*math.atan(0.5*height/fy)*180/math.pi 
	aspect = (width*fy)/(height*fx)
	near = 0.1 
	far = 100.0
	gluPerspective(fovy,aspect,near,far) 
	glViewport(0,0,width,height)
```

下面我们要将获得移除标定矩阵后的针孔照相机矩阵，并建立一个模拟视图

```python
def set_modelview_from_camera(Rt): 
	glMatrixMode(GL_MODELVIEW) 
	glLoadIdentity()
	Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
	R = Rt[:,:3] 
	U,S,V = np.linalg.svd(R) 
	R = np.dot(U,V) 
	R[0,:] = -R[0,:]
	t = Rt[:,3]
	M = np.eye(4) 
	M[:3,:3] = np.dot(R,Rx) 
	M[:3,3] = t
	M = M.T
	m = M.flatten()
	glLoadMatrixf(m)
```

接下来，我们需要做的事情是将图像作为背景添加进来，在OpenGL中，该操作可以通过创建一个四边形的方式来完成，该四边形作为整个视图。完成该操作的最简单方法是绘制出四边形，同时将投影和模拟视图矩阵重置，使得每一维的坐标范围在-1到1之间。

下面这个函数就可以实现将载入的一副图像，然后将其转换为OpenGL纹理，并将该纹理放置在四边形上

```python
def draw_teapot(size):
	glEnable(GL_LIGHTING) 
	glEnable(GL_LIGHT0) 
	glEnable(GL_DEPTH_TEST) 
	glClear(GL_DEPTH_BUFFER_BIT)
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0]) 
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0]) 
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0]) 
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0) 
	glutSolidTeapot(size)
```

最后将这些函数综合起来就是

```python
import math
import pickle
from pylab import *
from OpenGL.GL import * 
from OpenGL.GLU import * 
from OpenGL.GLUT import * 
import pygame, pygame.image 
from pygame.locals import *
from PCV.geometry import homography, camera
from PCV.localdescriptors import sift

def cube_points(c, wid):
    """ Creates a list of points for plotting
        a cube with plot. (the first 5 points are
        the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) #same as first to close plot
    
    # vertical sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    
    return array(p).T
    
def my_calibration(sz):
    row, col = sz
    fx = 2555*col/2592
    fy = 2586*row/1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5*col
    K[1, 2] = 0.5*row
    return K

def set_projection_from_camera(K): 
	glMatrixMode(GL_PROJECTION) 
	glLoadIdentity()
	fx = K[0,0] 
	fy = K[1,1] 
	fovy = 2*math.atan(0.5*height/fy)*180/math.pi 
	aspect = (width*fy)/(height*fx)
	near = 0.1 
	far = 100.0
	gluPerspective(fovy,aspect,near,far) 
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt): 
	glMatrixMode(GL_MODELVIEW) 
	glLoadIdentity()
	Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
	R = Rt[:,:3] 
	U,S,V = np.linalg.svd(R) 
	R = np.dot(U,V) 
	R[0,:] = -R[0,:]
	t = Rt[:,3]
	M = np.eye(4) 
	M[:3,:3] = np.dot(R,Rx) 
	M[:3,3] = t
	M = M.T
	m = M.flatten()
	glLoadMatrixf(m)

def draw_background(imname):
	bg_image = pygame.image.load(imname).convert() 
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)
	glMatrixMode(GL_MODELVIEW) 
	glLoadIdentity()

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
	glEnable(GL_TEXTURE_2D) 
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1)) 
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data) 
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST) 
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
	glBegin(GL_QUADS) 
	glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0) 
	glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0) 
	glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0) 
	glEnd()
	glDeleteTextures(1)


def draw_teapot(size):
	glEnable(GL_LIGHTING) 
	glEnable(GL_LIGHT0) 
	glEnable(GL_DEPTH_TEST) 
	glClear(GL_DEPTH_BUFFER_BIT)
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0]) 
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0]) 
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0]) 
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0) 
	glutSolidTeapot(size)

width,height = 1000,747
def setup():
	pygame.init() 
	pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF) 
	pygame.display.set_caption("OpenGL AR demo")    

# 计算特征
sift.process_image(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\code\bookcode\pcv-book-code-master\pcv-book-code-master\ch04\ar_cup\chapter4\book_frontal.JPG', 'im0.sift')
l0, d0 = sift.read_features_from_file('im0.sift')

sift.process_image(r'C:\Users\ZQQ\Desktop\advanced\study\computervision\code\bookcode\pcv-book-code-master\pcv-book-code-master\ch04\ar_cup\chapter4\book_perspective.JPG', 'im1.sift')
l1, d1 = sift.read_features_from_file('im1.sift')

# 匹配特征，并计算单应性矩阵
matches = sift.match_twosided(d0, d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)

model = homography.RansacModel()
H, inliers = homography.H_from_ransac(fp, tp, model)

# 计算照相机标定矩阵
K = my_calibration((747, 1000))

#投影第一幅图像上底部的三维点
cam1 = camera.Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# 底部正方形的点
box = cube_points([0, 0, 0.1], 0.1)
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))
box_trans = homography.normalize(dot(H,box_cam1))
cam2 = camera.Camera(dot(H, cam1.P))
A = dot(linalg.inv(K), cam2.P[:, :3])
A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

Rt=dot(linalg.inv(K),cam2.P)
 
setup() 
draw_background(r"C:\Users\ZQQ\Desktop\advanced\study\computervision\code\bookcode\pcv-book-code-master\pcv-book-code-master\ch04\ar_cup\chapter4\book_perspective.bmp") 
set_projection_from_camera(K) 
set_modelview_from_camera(Rt)
draw_teapot(0.05)

pygame.display.flip()
while True: 
	for event in pygame.event.get():
		if event.type==pygame.QUIT:
			sys.exit()

```

得到的实验结果是

![](https://github.com/zengqq1997/PCVch04/blob/master/result.jpg)

### 遇到的问题

1. 在安装OpenGL的时候，直接用pip install pyopengl语句，导致装错版本

   **解决方法**：执行pip install PyOpenGL-3.1.3b2-cp27-cp27m-win_amd64.whl即可完成安装

2. 在运行代码时候出现了如下错误

   ![](https://github.com/zengqq1997/PCVch04/blob/master/error.jpg)

   **解决方法**：将如下图中文件**freeglut.dll**删除掉即可解决

   ![](https://github.com/zengqq1997/PCVch04/blob/master/fangfa.jpg)

   

### 小结

解决了OpenGL的安装问题，以及代码运行出现的问题，实现简单的增强现实的例子
