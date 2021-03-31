import sys
try:
    #classes used in Segmentation task
    import numpy as np
    from matplotlib import pyplot as plt

    class model_parameters():
        def __init__(self, GT_img,data_img,obj_value):
            self.GT_img = GT_img
            self.data_img = data_img
            self.row, self.col = self.data_img.shape
            self.obj_value=obj_value

        def extract_Spatial_info(self): #method to extract spatial information of pixels of any class
            #considering background and chest as one class
            GT=np.zeros((self.row,self.col))
            for r in range(self.row):
                for c in range(self.col):
                    if (self.GT_img[r, c] <= 78):
                        GT[r,c]=78
                    else:
                        GT[r,c]=255
            #GT: img with pixel values:(78>> BG&chest),(255>>lung)

            object_pixels = [] #list that will filled with pixels location of specific class whose theshold=self.threshold
            for r in range(self.row):
                for c in range(self.col):
                    if (GT[r, c] == self.obj_value):
                        object_pixels.append((r, c))
            return object_pixels

        def object_pixel_values(self): #method to determine the values of pixels of certain class
            objpixels=model_parameters.extract_Spatial_info(self)
            objPixelValues = np.zeros((len(objpixels), 1))
            for i in range(len(objpixels)):
                objPixelValues[i]=(self.data_img[objpixels[i][0], objpixels[i][1]])
            return objPixelValues

        def get_parameters(self): #method to calculate the parameters of bayesian model
            objPixelValues=model_parameters.object_pixel_values(self) #array of pixel values of certain class

            mu=objPixelValues.mean()

            var=objPixelValues.var()

            prior = len(objPixelValues) / ((self.row * self.col))

            return mu,var,prior


    class model(): #class for modelling and plottting model

         def posteriori(self,mu,var,prior): #calculating posteriori
            qi=np.arange(0,256)
            try:
                posteriori=prior * (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(np.square(qi - mu)) / (2 * var))
            except ZeroDivisionError:
                print('ZeroDivisionError; may be the equation is incorrect, Check your equation')

            return posteriori

         def plotModel(self,obj1_prob,obj2_prob):#plotting Bayesian model
            titles=['TrainI','Train II','Train III','Train IV']
            gray_levels = np.arange(0, 256)
            fig=plt.figure(figsize=(15,15))
            plt.suptitle("Bayesian Model",fontsize=20)
            for i in range(4):
                plt.subplot(2,2,i+1)
                plt.title(titles[i])
                plt.plot(gray_levels, obj1_prob[i],'b',gray_levels, obj2_prob[i],'r',lw=1)
                plt.ylabel("Probability")
                plt.legend(['lung','chest'])
            fig.savefig('model.jpg')
            # return plt.show()

    class test_model(): #class to test model , calculate DSC of segmented image and plotting results

        def Bayesian_Decision(self,test_img,p_obj1,p_obj2): #test model
            r,c=test_img.shape
            segmented_obj1=np.zeros((r,c)) #initializing new array for segmented results(new img)

            for i in range(r):
                for j in range(c):
                    q = test_img[i, j]#pixel value

                    if (p_obj1[q] >= p_obj2[q]):#bayesian decision
                        segmented_obj1[i, j] = 255
                    else:
                        segmented_obj1[i, j] = 0
            return segmented_obj1

        def plotresults(self,originalImg,GT,segmented_obj1):#results plotting

            fig=plt.figure(figsize=(15, 15))
            plt.suptitle('Final Results',fontsize=20)
            for i in range(4):
                plt.subplot(4, 3, i * 3 + 1), plt.imshow(originalImg[i], 'gray')
                plt.title('originalImg'), plt.xticks([]), plt.yticks([])
                plt.subplot(4, 3, i * 3 + 2), plt.imshow(GT[i], 'gray')
                plt.title('GT '), plt.xticks([]), plt.yticks([])
                plt.subplot(4, 3, i * 3 + 3), plt.imshow(segmented_obj1[i], 'gray')
                plt.title('Segmented image'), plt.xticks([]), plt.yticks([])
            fig.savefig('results.jpg')
            # plt.show()
            # return figure

        def DSC(self,GT_img,segmented_obj): #Dice Similarity Calculation
            r,c=GT_img.shape
            GT=np.zeros((r,c)) #initializing new ground truth array(img) for updating pixel values
            #update ground truth pixel values to be=255(lung image),=0(any other pixel)
            for i in range(r):
                for j in range(c):
                    if(GT_img[i,j]<=78):
                        GT[i,j]=0
                    else:
                        GT[i,j]=255
            #initializing TP(true positive),FP(false positive),FN(false negative)
            TP=0
            FP=0
            FN=0
            for i in range(r):
                for j in range(c):
                    if segmented_obj[i, j] == GT[i, j]:
                        TP=TP+1
                    if (segmented_obj[i, j] - GT[i, j]) > 0:
                        FP = FP+1
                    if(segmented_obj[i, j] - GT[i, j]) < 0:
                        FN = FN+1
            try:
                DSC=(2*TP)/((2*TP)+FP+FN)
            except ZeroDivisionError:
                print('ZeroDivisionError; may be you have used incorrect images, Check your path & your image array')

            return DSC
except:
    print("Unexpected error:", sys.exc_info())

