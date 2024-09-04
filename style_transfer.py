import torch
from torchvision.transforms import v2
from ImageLoad import normalize_tensor_transform
from ImageLoad import ImageOps, ImageDataset, gram
from torch.autograd import Variable
from ImageTransform import ImageTransformNet
from PIL import Image

def style_transfer(model_path, source_image_path, style_image_path, output, gpu=1):
    if gpu == 1 and torch.backends.mps.is_available():
      device=torch.device("mps")
      use_cuda = True
      dtype = torch.float32
    else:
      dtype = torch.FloatTensor
      device="cpu"
    # content image
    img_transform_512 = v2.Compose([
            v2.Resize(512),                  # scale shortest side to image_size
            v2.CenterCrop(512),             # crop center image_size out
            v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),# turn image from [0-255] to [0-1]
            normalize_tensor_transform()])
            # normalize with ImageNet values


    #dtype = torch.FloatTensor
    source_image_ops = ImageOps(source_image_path,None)
    content = source_image_ops.load_image()
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)


    # load style model
    model_dict = torch.load(model_path)
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model = ImageTransformNet().type(dtype).to(device)
    style_model.load_state_dict(model_dict,False)

    # process input image
    stylized = style_model(content).to(device)
    output_image_ops = ImageOps(output, stylized.data[0])
    output_image_ops.save_image()
    size = (400,400)
    im1 = Image.open(source_image_path)
    im1 = im1.resize(size)
    print("Content Image: ")
    im1.show()
    im2 = Image.open(style_image_path)
    im2 = im2.resize(size)
    print("Style Image: ")
    im2.show()
    im = Image.open(output)
    im = im.resize(size)
    print("Output Image: ")
    im.show()