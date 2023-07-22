import argparse


def main():
    ap = argparse.ArgumentParser(description='Arguments', allow_abbrev=False)
    ap.add_argument("-d", "--dataset", default="data/dataset", help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", default="logs/h5/fashionnet.h5", help="path to output h5 model")
    ap.add_argument("-p", "--pb-model", default="logs/pb", help="path to output pb model")
    ap.add_argument("-g", "--h5_model", default="logs/pb", help="path to output pb model")
    ap.add_argument("-c", "--color", default="logs/label_map/color.pickle", help="path to output color label binarizer")
    ap.add_argument("-f", "--fig", default="logs/fig", help="base filename for generated plots")
    ap.add_argument("-l", "--cate", default="logs/label_map/category.pickle",
                    help="path to output category label binarizer")

    group = ap.add_argument_group("ckpt", "conf")
    group.add_argument("--ckpt_dir", default="logs/ckpt", help="path to output ckpt")
    args = vars(ap.parse_args())
    print(args)
    args_obj = ap.parse_args()
    print(args_obj.pb_model)  # --pb-model
    print(args_obj.h5_model)  # --h5_model
    print(args_obj.ckpt_dir)


if __name__ == '__main__':
    main()
