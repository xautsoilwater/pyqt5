from main import Ui_MainWindow
from PyQt5 import  QtWidgets,QtGui,QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox
import cv2
import numpy as np
from skimage import exposure
class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.OpenImage.triggered.connect(self.open)
        self.Exit.triggered.connect(self.exitP)
        self.f1=[]
        self.Sift.triggered.connect(self.msift)
        self.Surf.triggered.connect(self.msurf)
        self.actionRecognize.triggered.connect(self.recognize)
#        self.Daisy.triggered.connect(self.morb)
    def open(self):
        file,ok=QFileDialog.getOpenFileName(self,"打开",None,"*.jpg;;*.png;;*.tif;;*.bmp")
        self.f1.append(file)
        self.statusbar.showMessage(file)
    def exitP(self):
        result=QMessageBox.information(self,"Notice","Are you sure to exit",QMessageBox.StandardButtons(QMessageBox.Yes|QMessageBox.No))
        if result==QMessageBox.Yes:
            self.close()
    def msift(self):
        t1=cv2.getTickCount()
        im1=cv2.imread(self.f1[0],0)
        im2=cv2.imread(self.f1[1],0)
        sift=cv2.xfeatures2d.SIFT_create()
        kp1,des1=sift.detectAndCompute(im1,None)
        kp2,des2=sift.detectAndCompute(im2,None)
        im3=self.matchIMG(im1,im2,kp1,kp2,des1,des2)
        t2=cv2.getTickCount()
        time=(t2-t1)/cv2.getTickFrequency()
        self.statusbar.showMessage("时间是%s"%str(time))
        cv2.imshow('img',im3)
        cv2.waitKey(0)
        result=QMessageBox.information(self,"Notice","Do you want to save the final image",QMessageBox.StandardButtons(QMessageBox.Yes|QMessageBox.No))
        if result==QMessageBox.Yes:
            cv2.imwrite("Image.jpg",im3)
    def msurf(self):
        t1=cv2.getTickCount()
        im1=cv2.imread(self.f1[0],0)
        im2=cv2.imread(self.f1[1],0)
        surf=cv2.xfeatures2d.SURF_create()
        kp1,des1=surf.detectAndCompute(im1,None)
        kp2,des2=surf.detectAndCompute(im2,None)
        im3=self.matchIMG(im1,im2,kp1,kp2,des1,des2)
        t2=cv2.getTickCount()
        time=(t2-t1)/cv2.getTickFrequency()
        self.statusbar.showMessage("时间是%s"%str(time))
        cv2.imshow('img',im3)
        cv2.waitKey(0)
        result=QMessageBox.information(self,"Notice","Do you want to save the final image",QMessageBox.StandardButtons(QMessageBox.Yes|QMessageBox.No))
        if result==QMessageBox.Yes:
            cv2.imwrite("Image.jpg",im3)
    def recognize(self):
        face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face.load(r'E:\opencv-3.2.0\data\haarcascades\haarcascade_frontalface_default.xml')
        cap=cv2.VideoCapture(0)
        self.statusbar.showMessage("若想退出人脸识别视频流，则需要按下键盘的Esc键")
        while 1:
            ret,frame=cap.read()
            frame=exposure.adjust_gamma(frame,0.5)
            grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face.detectMultiScale(grayframe,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow('img',frame)
            k=cv2.waitKey(10)
            if k==27:
                break
        cv2.destroyAllWindows()
    def matchIMG(self,im1,im2,kp1,kp2,des1,des2):
        FLANN_INDEX_KDTREE=0
        index_p=dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        searth_p=dict(checks=50)
        flann=cv2.FlannBasedMatcher(index_p,searth_p)
        matches=flann.knnMatch(des1,des2,k=2)
        good =[]
        pts1=[]
        pts2=[]
        for i,(m,n) in enumerate(matches):
            if m.distance<0.6*n.distance:
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        pts1=np.float32(pts1)
        pts2=np.float32(pts2)
        F,mask=cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,0.01)
        pts1_1=pts1[mask.ravel()==1]
        pts2_2=pts2[mask.ravel()==1]
        pts2_new=pts2_2.copy()
        for i in range(len(pts2_2)):
            pts2_new[i,0]=pts2_new[i,0]+np.float32(im1.shape[1])
        def appendimage(im1,im2):
            r1=im1.shape[0]
            r2=im2.shape[0]
            if r1<r2:
                img=np.zeros((r2-r1,im1.shape[1]),np.uint8)
                im1_1=np.vstack((im1,img))
                im3=np.hstack((im1_1,im2))
            else:
                img=np.zeros((r1-r2,im2.shape[1]),np.uint8)
                im2_2=np.vstack((im2,img))
                im3=np.hstack((im1,im2_2))
            return im3
        im3=appendimage(im1,im2)     
        for i in range(len(pts1_1)):
            cv2.line(im3,tuple(pts1_1[i]),tuple(pts2_new[i]),255,2)
        return im3
if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    window=mywindow()
    window.show()
    app.exec_()

