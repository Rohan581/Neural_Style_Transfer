import torch
import os
from torchvision import transforms
from ImageLoad import normalize_tensor_transform
from ImageLoad import ImageOps, ImageDataset, gram
from torch.autograd import Variable
from torch.optim import Adam
from ImageTransform import ImageTransformNet
from VGG import Vgg16
from torch.utils.data import DataLoader
import time
from pathlib import Path
from torchvision.transforms import v2
from tqdm import tqdm


IMAGE_SIZE = 256
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 1
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7

cwd = Path(__file__).parent.absolute()

torch.utils.checkpoint.checkpoint = None

def train(style_image_path,dataset_path, visualize=True, gpu=1):
    if (gpu == 1 and torch.backends.mps.is_available()):
      device=torch.device("mps")
      use_cuda = True
      dtype = torch.float32
      #print("Current device: %d" %torch.cuda.current_device())
    else:
      dtype = torch.FloatTensor
      device="cpu"

    # visualization of training controlled by flag
    if (visualize):
        img_transform_512 = v2.Compose([
            v2.Resize(512),                  # scale shortest side to image_size
            v2.CenterCrop(512),             # crop center image_size out
            v2.ToImage(),
            v2.ToDtype(dtype),# turn image from [0-255] to [0-1]
            normalize_tensor_transform()]).to(device)
                          # normalize with ImageNet values


        amber_image_path = str(cwd) + "/content_imgs/amber.jpg"
        maine_image_path = str(cwd) + "/content_imgs/maine.jpg"

        amber_img_ops = ImageOps(amber_image_path, None)
        maine_img_ops = ImageOps(maine_image_path, None)

        testImage_amber = amber_img_ops.load_image()
        testImage_amber = img_transform_512(testImage_amber)
        testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

        testImage_maine = maine_img_ops.load_image()
        testImage_maine = img_transform_512(testImage_maine)
        testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False).type(dtype)

    # define network
    image_transformer = ImageTransformNet().type(dtype).to(device)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().type(dtype).to(device)

    # get training dataset
    dataset_transform = v2.Compose([
        v2.Resize(IMAGE_SIZE),           # scale shortest side to image_size
        v2.CenterCrop(IMAGE_SIZE),      # crop center image_size out
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),# turn image from [0-255] to [0-1]
        normalize_tensor_transform()]).to(device)
                                                # normalize with ImageNet values
    train_dataset = ImageDataset(root_dir=dataset_path, transform=dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)



    # style image
    style_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),# turn image from [0-255] to [0-1]
        normalize_tensor_transform()# normalize with ImageNet values
    ]).to(device)
    style_image_ops = ImageOps(style_image_path,None)
    style = style_image_ops.load_image()
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).to(device)
    style_name = os.path.split(style_image_path)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, x in enumerate(tqdm(train_loader)):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = x.to(device)
            x = Variable(x)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.data.item()

            # calculate total variation regularization (anisotropic version)
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()


            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                outputTestImage_amber = image_transformer(testImage_amber).cpu()
                amber_path = "visualization/%s/amber_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                amber_img_save_ops = ImageOps(amber_path,outputTestImage_amber.data[0])
                amber_img_save_ops.save_image()


                outputTestImage_maine = image_transformer(testImage_maine).cpu()
                maine_path = "visualization/%s/maine_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                maine_img_save_ops = ImageOps(maine_path, outputTestImage_maine.data[0])
                maine_img_save_ops.save_image()

                print("images saved")
                image_transformer.train()

    image_transformer.eval()

    if use_cuda:
        image_transformer.to(device)

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + ".model"
    torch.save(image_transformer.state_dict(), filename)
