import argparse
import os
from PIL import Image
import shutil


def resize_images(from_dir: str):
    os.makedirs(f"resized_{from_dir}", exist_ok=True)

    for identity_dir in os.listdir(from_dir):
        for filename in os.listdir(os.path.join(from_dir, identity_dir)):
            os.makedirs(os.path.join(f"resized_{from_dir}", identity_dir), exist_ok=True)

            # only process jpg files
            if filename.endswith(".jpg"):
                with Image.open(os.path.join(from_dir, identity_dir, filename)) as img:
                    resized_img = img.resize((112, 112))
                    resized_img.save(os.path.join(from_dir, identity_dir, filename))

    print(f"Removing {from_dir}...")
    shutil.rmtree(from_dir)

    print(f"Renaming resized_{from_dir} to {from_dir}...")
    os.rename(f"resized_{from_dir}", from_dir)


def rename_to_match_facenet(from_dir: str):
    for sub_folder in os.scandir(from_dir):
        if sub_folder.is_dir():
            new_dir_name = sub_folder.name.split("_")
            new_dir_name = new_dir_name[0] + "_".join(new_dir_name[1:])
            os.makedirs(os.path.join(f"new_{from_dir}", new_dir_name), exist_ok=True)

            for i, file_name in enumerate(os.scandir(sub_folder)):
                os.rename(file_name.path,
                          os.path.join(
                              os.path.join(f"new_{from_dir}", new_dir_name, new_dir_name + f"_{i + 1:04d}.jpg")))

    print(f"Removing {from_dir}...")
    shutil.rmtree(from_dir)

    print(f"Renaming new_{from_dir} to {from_dir}...")
    os.rename(f"new_{from_dir}", from_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to prepare the transferred images to run AdaFace.")

    parser.add_argument("--rename", default=False, required=False, action="store_true",
                        help="When True, the script will rename the dataset to match the Facenet dataset.")
    parser.add_argument("--resize", default=False, required=False, action="store_true",
                        help="When True, the script will resize the images to 112x112.")

    parser.add_argument("--bin", default=False, required=False, action="store_true",
                        help="When True, the script will binarize the test and val images.")
    parser.add_argument("--rec", default=False, required=False, action="store_true",
                        help="When True, the script will create the 'rec', 'lst', "
                             "and 'idx' files for the training split.")

    parser.add_argument("--all", default=False, required=False, action="store_true",
                        help="When True, the script will run all the steps.")

    args = parser.parse_args()

    splits = ["train", "test", "val"]

    if args.rename or args.all:
        for split in splits:
            print(f"Renaming {split} images...")
            rename_to_match_facenet(f"{split}")

    if args.resize or args.all:
        for split in splits:
            print(f"Resizing {split} images...")
            resize_images(f"{split}")

    if args.bin or args.all:
        for split in ["test", "val"]:
            print(f"Generating pairs.txt for {split}...")
            os.system(f"python gen_pairs_lfw.py --data-dir {split}/ --txt-file pairs.txt")
            print(f"Moving pairs.txt to {split}/...")
            os.system(f"mv pairs.txt {split}/")

            print(f"Creating {split}.bin...")
            os.system(f"python dataset2bin.py --data-dir {split}/ --image-size 112,112 --output {split}.bin")

    # if args.rec or args.all:
    #      print("Creating 'lst' files for the training split...")
    #     os.system("python im2rec.py --list --exts .jpg --recursive train ./train")
    #     print("Creating 'rec' and 'idx' files for the training split...")
    #     os.system("python im2rec.py train ./train")
