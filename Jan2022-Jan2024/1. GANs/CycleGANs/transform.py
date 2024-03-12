from CycleGANs import *
#=========================================================================
data_dir = {}
#=========================================================================
params = {
    'batch_size':1,
    'input_size':256,
    'resize_scale':286,
    'crop_size':256,
    'fliplr':True,
    # model params
    'num_epochs': 8,
    'decay_epoch':250,
    'ngf':32,   #number of generator filters
    'ndf':64,   #number of discriminator filters
    'num_resnet':6, #number of resnet blocks
    'lrG':0.0002,    #learning rate for generator
    'lrD':0.0002,    #learning rate for discriminator
    'beta1':0.5 ,    #beta1 for Adam optimizer
    'beta2':0.999 ,  #beta2 for Adam optimizer
    'lambdaA':10 ,   #lambdaA for cycle loss
    'lambdaB':10  ,  #lambdaB for cycle loss
}
#=========================================================================
G_A_new = Generator(3, params['ngf'], 3, params['num_resnet'])
G_B_new = Generator(3, params['ngf'], 3, params['num_resnet'])
#=========================================================================
transform = transforms.Compose([
    transforms.Resize(size=params['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_data_A = DatasetFromFolder(data_dir, subfolder='trainA', 
                                 transform=transform,
                                 resize_scale=params['resize_scale'], 
                                 crop_size=params['crop_size'], 
                                 fliplr=params['fliplr'])
train_data_loader_A = torch.utils.data.DataLoader(dataset=train_data_A, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True)
train_data_B = DatasetFromFolder(data_dir, subfolder='trainB', 
                                 transform=transform,
                                 resize_scale=params['resize_scale'], 
                                 crop_size=params['crop_size'], 
                                 fliplr=params['fliplr'])
train_data_loader_B = torch.utils.data.DataLoader(dataset=train_data_B, 
                                                  batch_size=params['batch_size'], 
                                                  shuffle=True)
#Load test data
test_data_A = DatasetFromFolder(data_dir, 
                                subfolder='testA', transform=transform)
test_data_loader_A = torch.utils.data.DataLoader(dataset=test_data_A, 
                                                 batch_size=params['batch_size'], 
                                                 shuffle=False)
test_data_B = DatasetFromFolder(data_dir, 
                                subfolder='testB', transform=transform)
test_data_loader_B = torch.utils.data.DataLoader(dataset=test_data_B, 
                                                 batch_size=params['batch_size'], 
                                                 shuffle=False)
# ------------------- Convert to 4d tensor (BxNxHxW) -----------------------
test_real_A_data = train_data_A.__getitem__(11).unsqueeze(0)
test_real_B_data = train_data_B.__getitem__(19).unsqueeze(0)
#=========================================================================
G_A_new.load_state_dict(torch.load('torch_cyclegan_A'))
G_B_new.load_state_dict(torch.load('torch_cyclegan_B'))
#=========================================================================
def test_view(k=-1):

    real_img = list(test_data_loader_A)[k][0]
    fake_B = G_A_new(real_img)
    recon_A = G_B_new(fake_B)
    img_transform = to_np(fake_B)
    imgs = [real_img, img_transform, recon_A]
    _, ax = plt.subplots(1, 2, figsize=(20, 7))

    real_img = to_np(real_img).squeeze()
    real_img = (((real_img - real_img.min()) * 255) / (real_img.max() - real_img.min())).transpose(1, 2, 0).astype(np.uint8)
    ax[0].imshow(real_img)
    ax[0].set_axis_off()

    img_transform = img_transform.squeeze()
    img_transform = (((img_transform - img_transform.min()) * 255) / (img_transform.max() - img_transform.min())).transpose(1, 2, 0).astype(np.uint8)
    ax[1].imshow(img_transform)
    ax[1].set_axis_off()
    
    plt.show()