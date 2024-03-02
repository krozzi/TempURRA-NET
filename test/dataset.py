from tempurranet.datasets.TUSimple import TuSimple
from config import tempurra_tusimple
import cv2
from tempurranet.util import hardware
import numpy as np
import time
import string

def interpolate_color(start_color, end_color, steps, current_step):
    if current_step < 0 or current_step >= steps:
        raise ValueError("Current step must be between 0 and steps-1")

    # Extract RGB components from start and end colors
    start_r, start_g, start_b = start_color
    end_r, end_g, end_b = end_color

    # Calculate interpolation factors for each color component
    factor = current_step / (steps - 1)
    interpolated_r = int(start_r + (end_r - start_r) * factor)
    interpolated_g = int(start_g + (end_g - start_g) * factor)
    interpolated_b = int(start_b + (end_b - start_b) * factor)
    return (interpolated_r/255, interpolated_g/255, interpolated_b/255)

# B G R
startclr = (0, 255, 0)
endclr = (0, 0, 255)

root = "../tempurranet/data/TUSimple"
split = "train_val"
dataset = TuSimple(root, split=split, processes=tempurra_tusimple.dataset['train']['processes'], cfg=tempurra_tusimple)
bruh = []
start=time.time()
for i in range(100):
    # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # for previous frames, the 1st index is the frame right BEFORE the frame to be predicted
    data = dataset.__getitem__(i)

    lanes = data['lanes']
    bruh.append(sum(len(x) for x in lanes))
    lanes = data['lanes']
    print(data)
    err_cnt = []

    # for some reason, unless the image is copied opencv sometimes errors while displaying it.
    # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
    cv2_image = np.transpose(data['img'].numpy(), (1, 2, 0)).copy()
    for idx, pt in enumerate(lanes):

        try:
            cv2.circle(cv2_image, (int(pt[0]), int(pt[1])), 1, interpolate_color(startclr, endclr, len(lanes), idx), 5)
        except Exception as e:
            print(e)
            err_cnt.append(e)
            tempurra_tusimple.logger.err("Error while drawing circle.")
            quit()
    tempurra_tusimple.logger.info(f"Image no.{i} | This many lane points: {sum(len(x) for x in lanes)} | {list(lanes)}")
    tempurra_tusimple.logger.info("this many errors: " + str(len(err_cnt)))

    show_img = cv2_image.copy()

    for image in data['prev_frames']:
        tmpimg = np.transpose(image.numpy(), (0, 1, 2)).copy()
        show_img = np.hstack((show_img, tmpimg))

    cv2.imshow("Image", show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
end=time.time()
print(bruh)
print(f"took {end-start} seconds to load 100 samples. thats around {100./(end-start)} samples/sec.")