#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from preprocess import preprocess
from keras.layers import Input
from discriminator import Discriminator
from keras.datasets import mnist
from random import randint
from copy import deepcopy
import argparse
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class Trainer:
    def __init__(self, width = 35, height= 1, channels = 1,  epochs =10, batch=10, checkpoint=50):
#        self.W = width
#        self.H = height
#        self.C = channels
#        self.EPOCHS = epochs
#        self.BATCH = batch
#        self.CHECKPOINT = checkpoint
#        self.model_type=model_type

#        self.LATENT_SPACE_SIZE = latent_size
#
#        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
#        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
#        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)
#
#        self.load_NSLKDD()
        
        
        
        self.EPOCHS = epochs
        self.BATCH = batch
        self.H = height
        self.W = width
        self.C = channels
        self.CHECKPOINT = checkpoint
#        self.LATENT_SPACE_SIZE = latent_size

#        self.X_train_B, self.X_train_A = self.load_data(train_data_path)
#        self.X_test_B, self.X_test_A  = self.load_data(test_data_path)
        self.X_train_B, self.X_train_A, self.X_test_B, self.X_test_A = self.load_DATASET()

        

        self.orig_A = Input(shape=(self.W, self.H, self.C))
        self.orig_B = Input(shape=(self.W, self.H, self.C))
        
        self.generator = Generator(height=self.H, width=self.W, channels=self.C)
        self.fake_A = self.generator.Generator(self.orig_B)
        
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        self.discriminator.trainable = False
        self.valid = self.discriminator.Discriminator([self.fake_A,self.orig_B])
            
#        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE)
#        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C)
        
        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid, self.fake_A]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs)



    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)

            number_of_batches = len(self.X_train_A)
        
            for b in range(number_of_batches):
                # Train Discriminator
                # Grab Real Images for this training batch
                starting_ind = randint(0, (len(X_train_A_temp)-1))
                real_images_raw_A = X_train_A_temp[ starting_ind : (starting_ind + 1) ]
                real_images_raw_B = X_train_B_temp[ starting_ind : (starting_ind + 1) ]

                # Delete the images used until we have none left
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_ind,(starting_ind + 1)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_ind,(starting_ind + 1)),0)

                batch_A = real_images_raw_A.reshape( 1, self.W, self.H, self.C )
                batch_B = real_images_raw_B.reshape( 1, self.W, self.H, self.C )

                # PatchGAN
#                y_valid = np.ones((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))
#                y_fake = np.zeros((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))

                y_valid = np.ones((1,)+(1, 9, 1))
                y_fake = np.zeros((1,)+(1, 9, 1))

                fake_A = self.generator.Generator.predict(batch_B)

                # Now, train the discriminator with this batch of reals
                discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A,batch_B],y_valid)[0]
                discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A,batch_B],y_fake)[0]
                full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

                generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B],[y_valid,batch_A])  
                print ('Epoch: '+str(int(e)),'Batch: '+str(int(b))+', [Full Discriminator :: Loss: '+str(full_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                self.writetocsv(full_loss, "./output/D_loss.csv",1,1)
                self.writetocsv(generator_loss, "./output/G_loss.csv",1,3)
                if  e > self.EPOCHS - 3 :
#                    label = str(e)+'_'+str(b)
#                    self.plot_checkpoint(label)
                    #noise = self.sample_latent_space(1)
                    #generated = self.generator.Generator.predict(noise)
                    self.writetocsv(fake_A, "./output/fake_examples.csv",self.C,self.H)

            print ('Epoch: '+str(int(e))+', [Full Discriminator :: Loss:'+str(full_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                        
                
        return

        

    def sample_latent_space(self, instances):
        return np.random.normal(0, 1, (instances,self.LATENT_SPACE_SIZE))

    def plot_checkpoint(self,e):
        filename = "./output/sample_"+str(e)+".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)
        
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H,self.W])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return



    def load_DATASET(self):
        #数据处理
        #args = parse_args()
        print("数据预处理中....")
        trainset = pd.read_csv('UNSW_NB15_training-set.csv')
        testset = pd.read_csv('UNSW_NB15_testing-set.csv')
        #去重
        #data = data.drop_duplicates()
        ##处理异常值
#
        #data = data.fillna(method='pad',axis=0)
        #X = data.iloc[:,1:-1]
        #y = data.iloc[:,-1]
        processor = preprocess()
        self.Worms,self.Shellcodes,self.Backdoor,self.Analysis,self.Normal,  self.Dos,self.Exploits,self.Fuzzers,self.Generic,self.Reconnaissance = processor.create_df(trainset, testset)
        print(self.Worms.head(3))
        self.Worms = self.Worms.drop(['attack_cat','label'], axis=1)
        self.Normal = self.Normal.drop(['attack_cat','label'], axis=1)
        self.Normal_train,self.Normal_test = train_test_split(self.Normal, test_size=.20, random_state=42)
        self.Worms_train,self.Worms_test = train_test_split(self.Worms, test_size=.20, random_state=42)
       
        self.Shellcodes = self.Shellcodes.drop(['attack_cat','label'], axis=1)
        self.Shellcodes_train,self.Shellcodes_test = train_test_split(self.Shellcodes, test_size=.20, random_state=42)
        
        self.Backdoor = self.Backdoor.drop(['attack_cat','label'], axis=1)
        self.Backdoor_train,self.Backdoor_test = train_test_split(self.Backdoor, test_size=.20, random_state=42)
        
        self.Analysis = self.Analysis.drop(['attack_cat','label'], axis=1)
        self.Analysis_train,self.Analysis_test = train_test_split(self.Analysis, test_size=.20, random_state=42)
        
        self.X_train = [self.Backdoor_train,self.Analysis_train]
        self.test_data = [self.Backdoor_test,self.Analysis_test]
        total_sample_size = 5000
        size = 50
        self.X, self.Y = self.get_data(size, total_sample_size, self.X_train)
        
        total_sample_size = 1000
        size = 200
        self.Tx, self.Ty = self.get_data(size, total_sample_size, self.test_data)
        return self.X[:,0],self.X[:,1],self.Tx[:,0],self.Tx[:,1]
    
    def load_data(self,data_path):
        listOFFiles = self.grabListOfFiles(data_path,extension="jpg")
        imgs_temp = np.array(self.grabArrayOfImages(listOFFiles))
        imgs_A = []
        imgs_B = []
        for img in imgs_temp:
            imgs_A.append(img[:,:self.H])
            imgs_B.append(img[:,self.H:])

        imgs_A_out = self.norm_and_expand(np.array(imgs_A))
        imgs_B_out = self.norm_and_expand(np.array(imgs_B))

        return imgs_A_out, imgs_B_out
    
    def get_data(self,size, total_sample_size, data_df):
        dim = 34
        count = 0
        
        x_geuine_pair = np.zeros([total_sample_size, 2, dim, 1]) 
        y_genuine = np.zeros([total_sample_size, 1])
        
        for i in range(0,len(data_df)):
            for j in range(int(total_sample_size/len(data_df))):
                ind1 = 0
                ind2 = 0
                
                while ind1 == ind2:
                    ind1 = np.random.randint(len(data_df[i]))
                    ind2 = np.random.randint(len(data_df[i]))
                
                data1 = data_df[i].iloc[ind1,:]
                data2 = data_df[i].iloc[ind2,:]
                
                x_geuine_pair[count, 0, :,0] = data1
                x_geuine_pair[count, 1, :,0] = data2
                
                y_genuine[count] = 1
                count += 1
    
#        count = 0
#        x_imposite_pair = np.zeros([total_sample_size, 2, dim, 1])
#        y_imposite = np.zeros([total_sample_size, 1])
#        
#        for i in range(int(total_sample_size/size)):
#            for j in range(size):
#                
#                while True:
#                    ind1 = np.random.randint(len(data_df))
#                    ind2 = np.random.randint(len(data_df))
#                    if ind1 != ind2:
#                        break
#                        
#                data1 = data_df[ind1].iloc[np.random.randint(len(data_df[ind1])),:]
#                data2 = data_df[ind2].iloc[np.random.randint(len(data_df[ind2])),:]
#                
#                x_imposite_pair[count, 0, :, 0] = data1
#                x_imposite_pair[count, 1, :, 0] = data2
#         
#                y_imposite[count] = 0
#                count += 1
#                
#        #now, concatenate, genuine pairs and imposite pair to get the whole data
#        X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)
#        Y = np.concatenate([y_genuine, y_imposite], axis=0)
        X = x_geuine_pair
        Y = y_genuine
    
        return X, Y
    
    def writetocsv(self, mtrx, flnm, row, col):
        """Save the samples for TDA with R (2nd notebook). We do not differentiate frauds from normal transactions
            保存生成的样本"""
        dtfrm = pd.DataFrame(np.array(mtrx).reshape(row, col))
        dtfrm.to_csv(flnm,mode='a+',sep=',', index=False, header=None)
        
        
#        test=pd.DataFrame(data=list)
#        print(test)
#        test.to_csv('e:/testcsv.csv',encoding='gbk')