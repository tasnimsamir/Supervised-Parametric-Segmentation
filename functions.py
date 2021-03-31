import sys
try:
    from PSBC import model_parameters,model,test_model
    import cv2
    from tqdm import tqdm

    class fnc():#class for setting up functions that implement final code
        def __init__(self,GT_files,data_files,lung=255,chest=78):
            self.GT_files=GT_files
            self.data_files=data_files
            self.lung=lung
            self.chest=chest
        # function to implement training
        def train(self,object, start, end):
            new_mu = 0
            new_var = 0
            new_prior = 0
            #reading images from its folder with cross validation concept
            for i in tqdm(range(start, end)):
                if i > (len(self.data_files) - 1):
                    k = i - (len(self.data_files) - 1)
                else:
                    k = i

                GTimg = cv2.imread(self.GT_files[k], 0)#reading ground truth image and convert it to gray scale
                # GTimg = cv2.medianBlur(GTimg, 5)
                DATimg = cv2.imread(self.data_files[k], 0) #reading data image and convert it to gray scale
                # DATimg = cv2.medianBlur(DATimg, 5)

                parameters = model_parameters(GTimg, DATimg, object) #initializing model_parameters class
                mu, var, prior = parameters.get_parameters()#returns model parameters

                #updating parameters in every iteration
                new_mu = new_mu + mu
                new_var = new_var + var
                new_prior = new_prior + prior


            return (new_mu / len(self.data_files) - 1), new_var / (len(self.data_files) - 1), new_prior / (len(self.data_files) - 1)

        # function to calculate probabilities
        def probabilities_calc(self,start, end):
            #calssI(lung)
            mu, var, prior = fnc.train(self,self.lung, start, end) #calling train method
            print("\nparameters of lung class \n")
            print('mean={}\t variance={}\t Prior={}\n'.format(mu,var,prior))

            mdl = model()#initializing model class
            p_lung = mdl.posteriori(mu, var, prior) #calculating posteriori of lung
            #classII(chest&background)
            mu, var, prior = fnc.train(self,self.chest, start, end)
            print("\nparameters of Chest class \n")
            print('mean={}\t variance={}\t Prior={}\n'.format(mu,var,prior))
            mdl = model()
            p_chest = mdl.posteriori(mu, var, prior) #calculating posteriori of lung

            return p_lung, p_chest
        # function to calculate DSC and test model
        def test_DSC(self,p_lung, p_chest, k):
            test_img = cv2.imread(self.data_files[k], 0) #reading test img
            # test_img = cv2.medianBlur(test_img, 5)
            GT_test = cv2.imread(self.GT_files[k], 0)#reading ground truth of test image
            tst = test_model()#initializing model class
            segmented_img = tst.Bayesian_Decision(test_img, p_lung, p_chest) #apply bayesian decision for segmentation
            DSC = tst.DSC(GT_test, segmented_img)#calculating DSC

            return test_img, GT_test, segmented_img, DSC

except:
    print("Unexpected error:", sys.exc_info())