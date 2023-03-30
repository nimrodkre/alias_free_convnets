import torch
from utils import get_args, get_model

WEIGHTS_PATH = r'C:\HUJI\lab_weiss\alias_free_convnets\alias_free_convnets\convnext_aps_tiny_f1_gelu_c_s2\checkpoint-best-ema.pth'

def main():
    args = get_args()
    model = get_model(args=args)
    checkpoint = torch.load(WEIGHTS_PATH,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    a = 5

if __name__ == "__main__":
    main()