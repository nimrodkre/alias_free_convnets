import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def shift_img(path, shifts, dims):
    img = Image.open(path)
    transform = transforms.ToTensor()
    tensor_img = transform(img)
    shifted_img = torch.roll(tensor_img, shifts=shifts, dims=dims)
    return shifted_img


def show_image(tensor_img):
    # Convert the tensor image to a NumPy array
    np_img = tensor_img.numpy()

    # Convert the NumPy array to a format that matplotlib can display
    np_img = np.transpose(np_img, (1, 2, 0))

    # Display the image using matplotlib
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()
    
def main():
    img = Image.open(r"C:\HUJI\lab_weiss\alias_free_convnets\images\ILSVRC2012_test_00099997.JPEG")
    transform = transforms.ToTensor()
    tensor_img = transform(img)
    # tensor_img = torch.Tensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.size[1], img.size[0], len(img.getbands())).permute(2, 0, 1).float().div(255)
    show_image(tensor_img=tensor_img)
    shifted_img = torch.roll(tensor_img, shifts=50, dims=1)
    show_image(shifted_img)
    x = 5


if __name__ == "__main__":
    main()