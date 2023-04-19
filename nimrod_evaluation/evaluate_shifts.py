import numpy as np
import utils
import torch
from timm.utils import accuracy

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        #
        shift0 = np.random.randint(-32, 32, size=2)
        shifted_inp0 = torch.roll(images, shifts=(shift0[0], shift0[1]), dims=(2, 3))
        # shift_subpix = IdealUpsample(2)(images)[:, :, 1::2, 1::2]

def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree