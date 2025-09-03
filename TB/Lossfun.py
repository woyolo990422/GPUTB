import torch

class Lossfunction(object):
    def __init__(self):
        # self.band_weight = 1
        pass

    def trainloss(self, eig, eig_ref, band_weight,e_range,dataset_id):
            MSEloss,MAEloss=0.0,0.0
            num_kn = 0
            
            for structure_index in range(len(eig)):
                e_ref = eig_ref[structure_index]
                e = eig[structure_index].to(device=e_ref.device)

                e_range=torch.as_tensor(e_range,device=e_ref.device)

                dataset_id_=dataset_id[structure_index]

                band_weight_=torch.as_tensor(band_weight[dataset_id_],dtype=torch.float32,device=e_ref.device)
                
                mask = (e_ref >= e_range[0]) & (e_ref <= e_range[1])
                e_ref = e_ref * mask
                e = e * mask

                diff=torch.abs(e-e_ref)
                diff2=(e-e_ref)**2

                num_kn += e.numel()
                MSEloss += torch.sum(diff2*band_weight_)
                MAEloss += torch.sum(diff*band_weight_)

            MAEloss = MAEloss/num_kn
            MSEloss = MSEloss/num_kn

            return    MAEloss**2+MSEloss


    def MSE_MAE_loss(self, eig, eig_ref):
        MSEloss, MAEloss = 0.0, 0.0

        e_ref = eig_ref
        e = eig.to(device=e_ref.device)
        num_kn = e.numel()
        MSEloss = torch.sum((e-e_ref)**2)
        MAEloss = torch.sum((e-e_ref).abs())

        MSEloss = MSEloss/num_kn  
        MAEloss = MAEloss/num_kn 


        return MSEloss.item(), MAEloss.item()
    
    def delta(self, energy_x,energy,width):
        x = -((energy_x - energy) / width)**2
        return torch.exp(x) / (torch.sqrt(torch.tensor(torch.pi)) * width)
    
    def get_dos_loss(self,para,npts,eigens,eigens_ref,device,need_dos=False):
        eigs_ref=eigens_ref[...,para.band_min[0]:para.band_max[0]]
        eigs=eigens[...,para.band_min[0]:para.band_max[0]]
        emin_ref,emax_ref,=eigs_ref.min(),eigs_ref.max()
        energy_x_ref = torch.linspace(emin_ref, emax_ref, npts, device=device).reshape(-1, 1)
        dos = torch.zeros(npts, device=device,dtype=torch.float64)
        dos_ref = torch.zeros(npts, device=device,dtype=torch.float64)

        for e_k, e_k_ref in zip(eigs, eigs_ref):
            dos += self.delta(energy_x_ref,e_k,para.dos_sigma).sum(axis=-1)
            dos_ref += self.delta(energy_x_ref,e_k_ref,para.dos_sigma).sum(axis=-1)

        dos/=eigs.shape[0]
        dos_ref/=eigs.shape[0]

        MAE_loss = torch.mean((dos - dos_ref).abs())
        MSE_loss = torch.mean((dos - dos_ref) ** 2)

        if need_dos:
            return dos,dos_ref,MSE_loss,MAE_loss
        else:
            return MSE_loss,MAE_loss
    

