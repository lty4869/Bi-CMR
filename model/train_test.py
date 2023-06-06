import torch
import model.models as ITnet
import torch.nn.functional as F
import eval.eval as eva
from loguru import logger
import time
import numpy as np
import os


def train(opt,train_loader,
              test_loader,
              dataset_loader,
              label_train_loader,
              ):

    #load model
    img_model,txt_model,label_model=ITnet.load_model(opt.mode_name,opt.code_length,opt.label_size,opt.input_size,opt.model)
    img_model.to(opt.device)
    txt_model.to(opt.device)
    label_model.to(opt.device)

    #parameters of model & optimizer of parameters
    img_params=list(img_model.parameters())
    txt_params=list(txt_model.parameters())
    label_params=list(label_model.parameters())

    img_optimizer = torch.optim.Adam(img_params,lr=opt.lr,betas =(0.5, 0.999))
    txt_optimizer=torch.optim.Adam(txt_params,lr=opt.lr,betas =(0.5, 0.999))
    label_optimizer = torch.optim.Adam(label_params, lr=opt.lr*10, betas =(0.5, 0.999))


    #Initialization
    loss=0
    avg_txt_img=0
    start = time.time()

    #Training epoch begin
    for epoch in range(opt.num_epochs):
        label_model.train()
        #multi-label network initialization (according to SSAH)
        for i,(labels, ind) in enumerate(label_train_loader):
            labels=labels.type(torch.FloatTensor).to(opt.device)
            _,label_hash_code,label_predict=label_model(labels)

            #initialization of Sij
            Sim = F.cosine_similarity(labels.unsqueeze(1), labels, dim=-1)
            Sim[Sim >= opt.lam] = 1
            Sim[Sim < opt.lam] = 0

            #according to SSAH
            theta = label_hash_code @ label_hash_code.t() / 2
            loss1 = (torch.log(1 + torch.exp(theta)) - Sim * theta).mean()
            loss2 = (-(torch.log(label_predict) * labels)).sum(1).mean()
            loss=loss1+0.5*loss2
            label_optimizer.zero_grad()
            loss.backward()
            label_optimizer.step()

        # training of modal and label networks
        img_model.train()
        txt_model.train()
        for i,(img, txt, labels, _, ind) in enumerate(train_loader):
            txt_trains = txt.to(opt.device)
            img_trains = img.to(opt.device)
            labels=labels.to(opt.device).type(torch.cuda.FloatTensor)

            #output of each model
            img_hash_code,img_predict, fi, fi_=img_model(img_trains)
            txt_hash_code,txt_predict, ft, ft_=txt_model(txt_trains)
            feature,label_hash_code,_=label_model(labels)

            #similarity matrix based on the concatenate of multi-label and multi-label hash code
            l = label_hash_code.sign()
            l[l < 0] = 0
            label_concat = torch.cat((labels, 0.01*l), dim=1)
            Sim = F.cosine_similarity(label_concat.unsqueeze(1), label_concat, dim=-1)
            Sim[Sim >= opt.lam] = 1
            Sim[Sim < opt.lam] = 0


            img_sim = F.cosine_similarity(img_hash_code.unsqueeze(1), img_hash_code, dim=-1)
            txt_sim = F.cosine_similarity(txt_hash_code.unsqueeze(1), txt_hash_code, dim=-1)
            img_txt_sim = F.cosine_similarity(img_hash_code.unsqueeze(1), txt_hash_code, dim=-1)

            #loss_ssr
            qi = F.log_softmax(fi_,dim=-1)
            qt = F.log_softmax(ft_, dim=-1)
            pi = F.softmax(fi,dim=-1)
            pt = F.softmax(ft, dim=-1)
            loss_ssr = F.kl_div(qi,pi,reduction='mean')+F.kl_div(qt,pt,reduction='mean')

            #loss_m2l
            eta = opt.eta
            a_img = torch.exp(img_sim)
            a_txt = torch.exp(txt_sim)
            # a_img = a_img ** 2
            # a_txt = a_txt ** 2
            # a_img = img_sim
            # a_txt = txt_sim
            a = eta * a_img + (1 - eta) * a_txt
            a_l = torch.exp(-F.cosine_similarity(label_hash_code.unsqueeze(1), label_hash_code, dim=-1))
            loss_m2l = (a_l* a).sum(1).mean()

            #loss_l2m
            a_label = F.cosine_similarity(label_hash_code.unsqueeze(1), label_hash_code, dim=-1)
            a_label = torch.exp(a_label)
            #a_label = a_label ** 2
            a_img_ = torch.exp(-img_sim)
            a_txt_ = torch.exp(-txt_sim)
            a_txt_img_ = torch.exp(-img_txt_sim)
            a_ =  a_img_ + a_txt_ + a_txt_img_
            loss_l2m= (a_label * a_).sum(1).mean()

            #correlation loss
            theta_imgtol=img_hash_code @ label_hash_code.t().detach()/2
            theta_ltoimg=label_hash_code.detach() @ img_hash_code.t()/2

            theta_txttol = txt_hash_code @ label_hash_code.t().detach()/2
            theta_ltotxt = label_hash_code.detach() @ txt_hash_code.t()/2

            theta_ltol = label_hash_code.detach() @ label_hash_code.t() / 2

            loss_cor=(torch.log(1 + torch.exp(theta_imgtol)) - Sim * theta_imgtol).mean()+\
                  (torch.log(1 + torch.exp(theta_ltoimg)) - Sim * theta_ltoimg).mean()+\
                  (torch.log(1 + torch.exp(theta_txttol)) - Sim * theta_txttol).mean()+\
                  (torch.log(1 + torch.exp(theta_ltotxt)) - Sim * theta_ltotxt).mean()+ \
                  (torch.log(1 + torch.exp(theta_ltol)) - Sim * theta_ltol).mean()

            #classification loss
            loss_class=-(torch.log(img_predict)*labels).sum(1).mean()-(torch.log(txt_predict)*labels).sum(1).mean()

            loss=  opt.alpha*loss_ssr + opt.beta*loss_l2m + opt.gamma*loss_m2l + loss_cor + loss_class

            img_optimizer.zero_grad()
            txt_optimizer.zero_grad()
            label_optimizer.zero_grad()
            loss.backward()
            img_optimizer.step()
            txt_optimizer.step()
            label_optimizer.step()


        #learning_rate
        if epoch % opt.iter == 0:
            for params in img_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)
            for params in txt_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)
            for params in label_optimizer.param_groups:
                params['lr'] = max(params['lr'] * 0.5, 1e-6)


        txt_to_img_maps=test(True,dataset_loader,test_loader,img_model,txt_model,opt.device,opt.code_length,opt.topK)
        img_to_txt_maps=test(False,dataset_loader,test_loader,img_model,txt_model,opt.device,opt.code_length,opt.topK)


        logger.info('[itr: {}][time:{:.4f}]'
                    '[total_loss: {:.4f}]'
                    '[I2Tmap: {:.4f}]'
                    '[T2Imap: {:.4f}]'.format(epoch + 1,
                                              time.time() - start,
                                              loss.item(),
                                              img_to_txt_maps,
                                              txt_to_img_maps))
        start=time.time()
        loss = 0

        if (txt_to_img_maps+img_to_txt_maps)/2>avg_txt_img:
            avg_txt_img=(txt_to_img_maps+img_to_txt_maps)/2
            #print the best average result
            print("avg_map:",avg_txt_img)
            #save the model
            txt_file_name = 'txt_net_{}_length_{}.t'.format(opt.dataname,opt.code_length)
            img_file_name = 'img_net_{}_length_{}.t'.format(opt.dataname,opt.code_length)
            torch.save(txt_model, os.path.join('result', txt_file_name))
            torch.save(img_model, os.path.join('result', img_file_name))
            txt_model.eval()
            img_model.eval()



#test function
def test(t2i,data_loader,test_loader,img_model,txt_model,device,code_length,topK):
    img_model.eval()
    txt_model.eval()
    if t2i: #text query images
        query_S = torch.FloatTensor(test_loader.dataset.S_data).to(device)
        query_code = code_(False,txt_model, test_loader, code_length, device).to(device)
        database_code=code_(True,img_model,data_loader, code_length, device).to(device)


    else: #image query texts
        query_S = torch.FloatTensor(test_loader.dataset.S_data).to(device)
        query_code = code_(True,img_model, test_loader, code_length, device).to(device)
        database_code = code_(False,txt_model, data_loader, code_length, device).to(device)
    #calculate the result of MAP
    meanAP = eva.MAP(query_code, database_code, query_S, device,topK)
    return meanAP

#generate binary hash codes
def code_(img,model,dataloader,code_length,device):
    with torch.no_grad():
        num=len(dataloader.dataset)
        code=torch.zeros([num,code_length])
        if img:
            for i,(trains,_,_,_,index) in enumerate(dataloader):
                trains=trains.to(device)
                outputs,_,_,_=model(trains)
                code[index,:]=outputs.sign().cpu()
        else:
            for i,(_,trains,_,_,index) in enumerate(dataloader):

                trains=trains.to(device)
                outputs,_,_,_=model(trains)
                code[index,:]=outputs.sign().cpu()
    return code




