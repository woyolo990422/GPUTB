import torch
import numpy as np
import sys
from TB.Parameters import Parameters
from TB.train_model import Train_Model
from TB.DatasetPreprocess import DatasetPrepocess
from torch_geometric.loader import DataLoader
from TB.Lossfun import Lossfunction
from utilities.Index import Index
import os
import random
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class Fitness(object):
    def __init__(self, para: Parameters, index: Index):
        self.para = para
        self.Index = index
        self.dtype = torch.float32
        self.device = self.para.nepin["device"]
        self.lr = self.para.nepin["lr"]
        self.lambda_2 = self.para.nepin["lambda_2"]
        self.decayfactor= self.para.nepin["decayfactor"]
        self.band_min = self.para.nepin["band_min"]
        self.band_max = self.para.nepin["band_max"]

        if len(self.para.dataset)==1:

            self.dataset = DatasetPrepocess(para,self.Index).dataset
            
            total_data_num=len(self.dataset)

            # if self.para.nepin["prediction"]:
            #     batch_size=total_data_num
            #     self.para.batch_size=batch_size
            # else:
            #     batch_size=self.para.batch_size
            self.data_batch=[]
            batch_size=self.para.batch_size

            # if batch_size>=total_data_num:
            #     self.data_batch.append(DataLoader(self.dataset,batch_size=batch_size))
            # elif batch_size<total_data_num:
            batch_times=total_data_num//batch_size
            if total_data_num%batch_size!=0:
                batch_times+=1
            for index in range(batch_times):
                start=batch_size*index
                end=batch_size*(index+1)
                if  total_data_num%batch_size==0 or index!=batch_times-1:
                    self.data_batch.append(DataLoader(self.dataset[start:end],batch_size=para.batch_size))
                elif total_data_num%batch_size!=0 and index==batch_times-1:
                    self.data_batch.append(DataLoader(self.dataset[start:],batch_size=para.batch_size))

            self.all_data=[]
            for data_index,data in enumerate (self.data_batch):
                
                for data2 in data:
                    
                    self.all_data.append(data2)
            self.mix_num=0

        else:
            
            self.dataset = DatasetPrepocess(para,self.Index).dataset
            self.data_batch=[]
            for data_index,data in enumerate (self.dataset):
                self.data_batch.append(data)

            self.all_data=[]
            for data_index,data in enumerate (self.data_batch):
                 
                    self.all_data.append(data)

            self.mix_num=len(self.all_data)

        self.lossfunction = Lossfunction()

        # self.indices = list(range(len(self.all_data)))

        # random.shuffle(self.indices)

        # self.all_data = [self.all_data[i] for i in self.indices]

        self.model = Train_Model(self.para, self.Index,self.all_data)

        self.model.to(self.device)
        
        print("Backward Parameters Num:",sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.optimizer = getattr(torch.optim, self.para.nepin["optimizer"])( filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr, weight_decay=self.lambda_2)
        
        if self.para.nepin["lr_method"]=="adapt":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=self.decayfactor,patience=50,threshold=0.001)
        elif self.para.nepin["lr_method"]=="exp":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.decayfactor)

        self.band_weight=self.para.nepin["band_weight"]

    def compute(self, generation, time_str):
        test_set_num = max(int(len(self.all_data) * self.para.test_ratio), 1)

        train_set_num=(len(self.all_data) - test_set_num)

        test_MSE_ave,test_MAE_ave,train_MSE_ave,train_MAE_ave=0.0,0.0,0.0,0.0

        for index in range(len(self.all_data)):
            data = self.all_data[index]
            self.eig_ref = data.eigs_ref
            data = data.to(device=self.device)

            if index < train_set_num or test_set_num == 1:
                
                self.model.train()
                
                eigs_list = self.model(data, index)
                eig, eig_ref = self.eigs_preprocess(eigs_list, self.eig_ref, data.dataset_id)
                
                train_loss = self.lossfunction.trainloss(eig, eig_ref, self.band_weight,self.para.e_range, data.dataset_id).norm()
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.para.nepin["lr_method"] == "adapt":
                    self.lr_scheduler.step(train_loss)
                elif self.para.nepin["lr_method"] == "exp":
                    self.lr_scheduler.step()

                if generation % self.para.nepin["step_interval"] == 0:
                    train_MSEloss, train_MAEloss = self.lossfunction.MSE_MAE_loss(eig, eig_ref)
                    train_MSE_ave+=train_MSEloss
                    train_MAE_ave+=train_MAEloss
                    print(f"{generation}  {train_loss.item():.6f} {train_MSEloss:.6f} {train_MAEloss:.6f} {self.lr_scheduler.get_last_lr()[0]:.6f} {time_str} TRAIN_ID_{index}")
                    sys.stdout.flush()
            else:
                if generation % self.para.nepin["step_interval"] == 0:
                    eigs_list = self.model(data, index)
                    eig, eig_ref = self.eigs_preprocess(eigs_list, self.eig_ref, data.dataset_id)
                    test_MSEloss, test_MAEloss = self.lossfunction.MSE_MAE_loss(eig, eig_ref)
                    test_MSE_ave+=test_MSEloss
                    test_MAE_ave+=test_MAEloss
                    print(f"test loss  {test_MSEloss:.6f} {test_MAEloss:.6f} {time_str} TEST_ID_{index-train_set_num}")
                    sys.stdout.flush()


        if generation % self.para.nepin["step_interval"] == 0:
            print(f"============================== {generation} {self.lr_scheduler.get_last_lr()[0]:.6f} AVE {train_MSE_ave/max(train_set_num,1):.6f} {train_MAE_ave/max(train_set_num,1):.6f} {test_MSE_ave/test_set_num:.6f} {test_MAE_ave/test_set_num:.6f} ==============================")
            sys.stdout.flush()

    def eigs_preprocess(self,eigs_list,eigs_ref_list,dataset_id):
        eigs_ref_list=eigs_ref_list.unsqueeze(0).reshape(eigs_list.shape[0],eigs_list.shape[1],-1)
        band_min=torch.as_tensor(self.band_min,dtype=torch.int32)[dataset_id[0]]
        band_max=torch.as_tensor(self.band_max,dtype=torch.int32)[dataset_id[0]]
        eig_cut=eigs_list[:,:, band_min:band_max]
        eig_ref_cut=eigs_ref_list[:,:, band_min:band_max]
        return eig_cut, eig_ref_cut

    def save_with_incremental_filename(self,directory, file_name, format="svg", transparent=True):
        """
        保存文件，如果文件名已存在，则自动递增编号。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)  # 如果目录不存在，创建它
        
        base, ext = os.path.splitext(file_name)
        save_path = os.path.join(directory, file_name)
        counter = 1
        while os.path.exists(save_path):
            file_name = f"{base}{counter}{ext}"
            save_path = os.path.join(directory, file_name)
            counter += 1
        
        plt.savefig(save_path, format=format, transparent=transparent)
        return save_path


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for data_index,data in enumerate(self.all_data):
                eigs_list = self.model(data,data_index)
                self.eig_ref = data.eigs_ref
                eig, eig_ref = self.eigs_preprocess(eigs_list,self.eig_ref,data.dataset_id)

                for i in range(eig.shape[0]):
                    test_MSEloss, test_MAEloss = self.lossfunction.MSE_MAE_loss(eig[i], eig_ref[i])
                    #mse_dos,mae_dos=self.lossfunction.get_dos_loss(self.para,10000,eig[i],eig_ref[i],self.device)
                    print(f"prediction  MSE-EIG:  {test_MSEloss:.5f}, MAE-EIG:  {test_MAEloss:.5f}")
                    sys.stdout.flush()
                    #print(f"prediction  MSE-DOS:  {mse_dos:.5f}, MAE-DOS:  {mae_dos:.5f}")
                
                if len(eig) == 1:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                    # 绘制图像
                    ax[0].scatter(eig.to("cpu").flatten(), eig_ref.to("cpu").flatten(), s=1)
                    ax[1].plot(range(eig[i].shape[0]), eig[i].to("cpu"), color='red', linewidth=2)
                    ax[1].plot(range(eig_ref[i].shape[0]), eig_ref[i].to("cpu"), color='grey',linestyle='--',dashes=(5, 5),  linewidth=2)
                    ax[1].tick_params(direction='in')
                    ax[1].tick_params(axis='x', length=0)
                    # ax[1].set_ylim(-15, 20)
                    ax[1].set_xlim(0, eig_ref.shape[1] - 1)

                    # 保存路径和文件名
                    save_directory = self.para.nepin["error_figures_path"]
                    file_name = "figure.svg"

                    # 保存文件
                    saved_path = self.save_with_incremental_filename(save_directory, file_name)
                    print(f"Figure saved at: {saved_path}")

                    # 显示图像
                    plt.show()
                    
                else:
                    for i in range(len(eig)):
                        fig,ax = plt.subplots(1, 2, figsize=(12, 6))
                        ax[0].scatter(eig[i].to("cpu").flatten(),eig_ref[i].to("cpu").flatten(), s=1)
                        ax[1].plot(range(eig[i].shape[0]), eig[i].to("cpu"), color='red', linewidth=2)
                        ax[1].plot(range(eig_ref[i].shape[0]), eig_ref[i].to("cpu"), color='grey',linestyle='--',dashes=(5, 5),  linewidth=2)

                        if self.para.nepin["error_figures_path"] is not None:
                            save_directory = self.para.nepin["error_figures_path"]
                            file_name = f'error_{i}.svg'
                            save_path = os.path.join(save_directory, file_name)
                            plt.savefig(save_path, format='svg', bbox_inches='tight')
                            plt.close(fig) 
                        else:
                            plt.savefig(f'error_{i}.svg', format='svg', bbox_inches='tight')
                            plt.close(fig) 


        save_directory = self.para.nepin["error_figures_path"]

        np.savetxt(os.path.join(save_directory, "scatter_x.txt"), eig.to("cpu").flatten().numpy())
        np.savetxt(os.path.join(save_directory, "scatter_y.txt"), eig_ref.to("cpu").flatten().numpy())
        np.savetxt(os.path.join(save_directory, "scatter_disty.txt"), (eig.to("cpu")[0].numpy()-eig_ref.to("cpu")[0].numpy()).flatten())
        np.savetxt(os.path.join(save_directory, "line_x1.txt"), range(eig.shape[1]))
        np.savetxt(os.path.join(save_directory, "line_y1.txt"), eig.to("cpu")[0].numpy())
        np.savetxt(os.path.join(save_directory, "line_x2.txt"), range(eig_ref.shape[1]))
        np.savetxt(os.path.join(save_directory, "line_y2.txt"), eig_ref.to("cpu")[0].numpy())

    def predict_large(self):
        self.model.eval()
        with torch.no_grad():
            for data_index,data in enumerate(self.all_data):
                eigs_list = self.model(data,data_index)
                self.eig_ref=data.eigs_ref
                eig, eig_ref = self.eigs_preprocess(eigs_list,self.eig_ref,data.dataset_id)

                for i in range(eig.shape[0]):
                    test_MSEloss, test_MAEloss = self.lossfunction.MSE_MAE_loss(eig[i], eig_ref[i],self.para)
                    print(f"prediction  MSE-EIG:  {test_MSEloss:.5f}, MAE-EIG:  {test_MAEloss:.5f}")


            fig,ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].scatter(eig.to("cpu").flatten(),eig_ref.to("cpu").flatten(), s=1)
            ax[1].plot(range(0, eig_ref.shape[1]),eig_ref.to("cpu")[0],  marker='o', markersize=4,markeredgewidth=1, color='gray', linewidth=0)
            ax[1].plot(range(eig.shape[1]), eig.to("cpu")[0],  color='red', linewidth=1.5)
            ax[1].tick_params(direction='in')
            ax[1].tick_params(axis='x', length=0)
            # ax[1].set_ylim(-15, 20)
            ax[1].set_xlim(0, eig_ref.shape[1]-1)
            plt.savefig("figure.svg", format="svg", transparent=True)
            plt.show()
                


        save_directory = self.para.nepin["error_figures_path"]

        np.savetxt(os.path.join(save_directory, "scatter_x.txt"), eig.to("cpu").flatten().numpy())
        np.savetxt(os.path.join(save_directory, "line_x1.txt"), range(eig.shape[1]))
        np.savetxt(os.path.join(save_directory, "line_y1.txt"), eig.to("cpu")[0].numpy())
    