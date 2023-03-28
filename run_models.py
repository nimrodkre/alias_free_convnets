import torch
from utils import get_args, get_model

WEIGHTS_PATH = r'C:\HUJI\lab_weiss\alias_free_convnets\alias_free_convnets\convnext_afc_tiny_ideal_up_poly_per_channel_scale_7_7_chw2_stem_mode_lpf_poly_cutoff0.75\checkpoint-best-ema.pth'

def main():
    args = get_args()
    model = get_model(args=args)
    checkpoint = torch.load(WEIGHTS_PATH,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    a = 5

if __name__ == "__main__":
    main()