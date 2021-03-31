import sys

try:
    from PSBC import model,test_model
    import glob
    from functions import fnc
    import time

    class impl():
        def __init__(self):
            #loading images from its folder
            self.GT_files = glob.glob(r'Supervised project\GT\*.bmp')
            self.data_files = glob.glob(r'Supervised project\data\*.bmp')

        # __________________________________________________________________________________
        #start training
        def im(self):
            func=fnc(self.GT_files,self.data_files) #initializing fnc class
            Dsc=[]
            GT=[]
            original=[]
            segmentedImg=[]
            pLung=[]
            pChest=[]
            for i in range(len(self.data_files)):
                #training using k fold concept (cross validation)
                print('\ntraining trial {} ..............\n'.format(i+1))
                start=i
                end=(len(self.data_files)-1)+i
                k=end #determine test image
                if k>(len(self.data_files)-1):
                    k=end-(len(self.data_files))

                p_lung,p_chest=func.probabilities_calc(start,end) #posteriori of bayesian model calculation
                pLung.append(p_lung)
                pChest.append(p_chest)

                originalImg,Gt,segmented_img,DSC=func.test_DSC(p_lung,p_chest,k)  #test new image and DSC calculation
                Dsc.append(DSC)
                original.append(originalImg)
                GT.append(Gt)
                segmentedImg.append(segmented_img)

            #plotting results
            mdl=model() #initializing model class
            mdl.plotModel(pLung,pChest)

            tst=test_model() #initializing test_model class
            tst.plotresults(original,GT,segmentedImg)

            #final Dice similarity
            print('\nDSC of each training trial= {}'.format(Dsc))
            print('Average DSC = {}\n'.format(sum(Dsc)/len(self.data_files)))
            print('Best DSC= {}\t at training trial number= {}\t'.format(max(Dsc),(Dsc.index(max(Dsc))+1)))
            # return fig
    start=time.time()
    imm=impl()
    imm.im()
    end=time.time()
    training_duration=end-start
    print("total training duration= {} seconds".format(training_duration))
except:
    print("Unexpected error:", sys.exc_info())