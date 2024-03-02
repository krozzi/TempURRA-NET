import os
import json
import argparse
import ast
import logging
import math
from tqdm import tqdm

TEST_SET = "test_label_new.json"
TEST_SET_ALT = "test_label.json"

TRAIN_SET = "train_0313_0601.json"
VAL_SET = "validation_0531.json"
TRAIN_VAL_SET = "train_val.json"



def interpolate_points(points, N):

    if N < 0:
        return points[:-abs(N)]


    interp = points.copy()
    i = 0
    pt_len = len(interp)

    while N > 0:

        if i >= pt_len:
            pt_len = len(interp)
            i = 0
            continue

        nxt_pt = interp[i + 1]
        cur_pt = interp[i]

        interp.insert(i + 1, [
            cur_pt[0] + 0.5 * (nxt_pt[0] - cur_pt[0]),
            cur_pt[1] + 0.5 * (nxt_pt[1] - cur_pt[1]),
        ])

        i += 2
        N -= 1

    return interp


def get_data(dataset, raw_file: str = ""):
    for l_data in dataset:
        if l_data['raw_file'] == raw_file:
            return l_data
        elif "test_set/" + l_data['raw_file'] == raw_file:
            return l_data
    return None


def reorder_json(t_args: argparse.Namespace, save_dir: str, json_dat: str, past_frames: int = 5, spacing: int = 1,
                 save_name: str = "ordered.json", desired_net_points: int = int(360 / 2)) -> None:
    """
    Reorders all the entries in the TUSimple dataset's json files to ensure it is time-series correct.
    :param t_args: Argument containing root dir of the TUSimple dataset.
    :param save_dir: Directory in which to save the file in.
    :param json_dat: Name of JSON file containing stuff you want to reorder.
    :param past_frames: Number of frames to collect.
    :param spacing: How many frames between each of the frames.
    :param save_name: Name of the final output json.
    :param desired_net_points: The total number of points you would like to have. Consider only (x, y) pairs.
    :return: None
    """

    json_root = os.path.join(t_args.root, "raw")
    data_path = os.path.join(json_root, json_dat)

    clips_pth = os.path.join(t_args.root, "clips")
    clips_dir = os.listdir(clips_pth)
    clips_dir_path = list(map(lambda x: os.path.join(clips_pth, x), clips_dir))

    past_frame_ints = list(range(19, 20-(spacing*past_frames)-1, -spacing))
    for frame_int in past_frame_ints:
        if frame_int < 0:
            raise ValueError(f"Past frames and spacing dont work out. {past_frame_ints}")

    write_data = []

    with open(data_path, "r") as lane_json:
        dataset = lane_json.read().splitlines()
        dataset = [json.loads(x) for x in dataset]

        for path in clips_dir_path:
            clip_group_pth = os.listdir(path)
            clip_group_dir = list(map(lambda x: os.path.join(path, x), clip_group_pth))

            for clip_group in clip_group_dir:
                clip_frame_pth = os.listdir(clip_group)
                clip_frame_dir = list(map(lambda x: os.path.join(clip_group, x), clip_frame_pth))

                clip_frame_dir_rel = [x.replace("\\", "/")[path.find('clips'):] for x in clip_frame_dir]

                for clip_pth, clip_form_pth in tqdm(zip(clip_frame_dir, clip_frame_dir_rel)):
                    if "20.jpg" in clip_form_pth:
                        frame_data = get_data(dataset, clip_form_pth)
                        if frame_data is None:
                            continue
                        lanes = frame_data['lanes']
                        diff_count = []
                        new_lanes = [[] for _ in range(len(lanes))]

                        h_samples = frame_data['h_samples']

                        for idx, lane in enumerate(lanes):
                            for x, y in zip(lane, h_samples):
                                if x == -2 or y == -2:
                                    continue
                                new_lanes[idx].append([x, y])

                        new_lanes = [x for x in new_lanes if not len(x) <= 0 and not len(x) == 1]
                        points_per_lane = math.ceil(desired_net_points / len(new_lanes))

                        for lane in new_lanes:
                            diff_count.append(points_per_lane - len(lane))

                        new_final_lanes = []
                        for idx, diff in enumerate(diff_count):
                            new_final_lanes.append(interpolate_points(new_lanes[idx], diff))

                        prev_frames = [clip_form_pth.replace("20.jpg", f"{x}.jpg") for x in past_frame_ints]
                        data = {"lanes": new_final_lanes,
                                "h_samples": h_samples,
                                "prev_frames": prev_frames,
                                "raw_file": clip_form_pth}
                        write_data.append(data)

    with open(os.path.join(t_args.root, save_dir, save_name), "w") as output:
        output.writelines(map(lambda x: str(x).replace("\'", "\"") + "\n", write_data))
        output.close()

    print("Finished")

def gen_json(t_args: argparse.Namespace) -> None:
    """
    Reorders all the entries in the TUSimple dataset's json files to ensure it is time-series correct.
    :param t_args: Argument containing root dir of the TUSimple dataset
    :return: None
    """
    save_dir = os.path.join(t_args.root, t_args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # comment out if test
    reorder_json(t_args, t_args.save_dir, TRAIN_VAL_SET, past_frames=5, spacing=1, save_name="ord_train_val.json")
    reorder_json(t_args, t_args.save_dir, TRAIN_SET, past_frames=5, spacing=1, save_name="ord_train.json")
    reorder_json(t_args, t_args.save_dir, VAL_SET, past_frames=5, spacing=1, save_name="ord_val.json")

    # comment out if train
    # reorder_json(t_args, t_args.save_dir, TEST_SET_ALT, past_frames=5, spacing=1, save_name="ord_test.json")


if __name__ == '__main__':

    """
    Roots
    ../tempurranet/data/TuSimple/execution
    ../tempurranet/data/TuSimple/test
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--root',
                        required=True,
                        help='The root of the TuSimple dataset')
    parser.add_argument('--save_dir',
                        type=str,
                        default='fixed',
                        help='The root of the TuSimple dataset')
    args = parser.parse_args()
    gen_json(args)
