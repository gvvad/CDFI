import argparse
import torch
# from torchvision import transforms
from models.cdfi_adacof import CDFI_adacof
from ffHelper import FFSource, FFDestination, frameRateMul
import time

parser = argparse.ArgumentParser(description='FFmpeg 2x frames interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')

parser.add_argument('--kernel_size', type=int, default=11)
parser.add_argument('--dilation', type=int, default=2)

parser.add_argument("-s", '--source', dest="src", type=str, required=True)
parser.add_argument("-d", '--destination', dest="dst", type=str, required=True)
parser.add_argument("-ffsp", '--ff_src_param', dest="ffsp", type=str)
parser.add_argument("-ffdp", '--ff_dst_param', dest="ffdp", type=str)

args = parser.parse_args()

def rgbToTensor(frame_rgb):
    return torch.from_numpy(frame_rgb / 255.0).float().permute(2, 0, 1).unsqueeze(0)

def tensorToRgb(frame):
    return (frame.squeeze().clamp(0.0, 1.0) * 255).byte().permute(1, 2, 0).cpu().numpy()

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_id)
    
    print('Model initialization...')
    model = CDFI_adacof(args).float().cuda()
    print('Loading weights...')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    src = FFSource(args.src, param=args.ffsp.split(" ") if args.ffsp is not None else [])
    with FFDestination(args.dst,
                        src.streamInfo["width"],
                        src.streamInfo["height"],
                        frameRateMul(src.streamInfo["r_frame_rate"], 2),
                        param=args.ffdp.split(" ") if args.ffdp is not None else []) as dest:
            
        with torch.no_grad():
            n = 0
            frame_a = None
            for frame in src:
                if frame_a is None:
                    dest.putData(frame)
                    frame_a = frame
                    continue
                n += 1
                time_stamp = time.time()
                print(f"frame: {n} processing...")

                a = rgbToTensor(frame_a).cuda()
                b = rgbToTensor(frame).cuda()
                iframe = model(a, b)
                
                dest.putData(tensorToRgb(iframe))
                dest.putData(frame)
                print(f"frame: {n} done proc. Time: {(time.time()-time_stamp):.2f}")
                frame_a = frame
        
        print(f"Frames total:{n}")
