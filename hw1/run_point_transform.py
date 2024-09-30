import cv2
import numpy as np
import gradio as gr
import math
# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    n=source_pts.shape[0]
    d=np.zeros((n,1))
    if n==1:
        dist=pow(pow(source_pts[0][0]-target_pts[0][0],2)+pow(source_pts[0][1]-target_pts[0][1],2),1)
        d[0]=dist
    if n>1:
        for i in range(0,n):
            d[i]=1e10
            for j in range(0,n):
                if j!=i:
                    d[i]=min(pow(pow(target_pts[i][0]-target_pts[j][0],2)+pow(target_pts[i][1]-target_pts[j][1],2),1),d[i])

    R=np.zeros((n,n))
    b=np.zeros((n,2))
    for i in range(0,n):
        b[i]=source_pts[i]-target_pts[i]
        for j in range(0,n):
            R[i][j]=1/(pow(target_pts[i][0]-target_pts[j][0],2)+pow(target_pts[i][1]-target_pts[j][1],2)+d[j])

    alpha = np.linalg.solve(R, b)
    print(target_pts)
    print(warped_image.shape)
    for i in range(0,warped_image.shape[0]):
        for j in range(0,warped_image.shape[1]):
            idx=np.array((2,1))
            for k in range(0,n):
                idx=idx+alpha[k]*1/(pow(target_pts[k][0]-j,2)+pow(target_pts[k][1]-i,2)+d[k])
            idx=idx+[j,i]
            idx[0]=min(image.shape[1]-1,idx[0])
            idx[1] = min(image.shape[0] - 1, idx[1])
            idx[0]=max(0,idx[0])
            idx[1] = max(0, idx[1])


            warped_image[i][j]=image[math.floor(idx[1])][math.floor(idx[0])]
            if i==target_pts[0][1]:
                if j==target_pts[0][0]:
                    print(idx)
                    print(source_pts[0])

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
