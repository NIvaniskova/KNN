## Preprocessing dataset for model training
The names of the files and directories of the dataset need to follow a convention of AdaFace (presented by FaceNet) 
which WebFace doesn't.
We also made a little mistake when transferring the style and resized the images to 512x512 pixels, 
thus resizing them to 112x112 is necessary.

Both of these problems can be solved by running the script `preprocess_dataset.py` in this directory. 
The script expects `train`, `val`, and `test` directories in the same path. It is recommended to run the 
`--rename` and `--resize` flags separately. Both remove the original directories and create new ones with the same name.

## Binary files for testing and training
[Talgrin/preparing_data](https://github.com/Talgin/preparing_data) repo 
and [tutorial](https://github.com/deepinsight/insightface/issues/791#issuecomment-511108136) were of help when creating 
the binary files. It includes scripts to create `pairs.txt` and convert such folders into `.bin` files. Some scripts 
like `facenet.py` and `lfw.py` come from the [Facenet](https://github.com/davidsandberg/facenet.git) repo.

## Train labels, ids and records
For the converting of the training split one can follow and use the `im2rec.py` script from 
[MXnet](https://mxnet.apache.org/versions/1.9.1/api/faq/recordio).

For simplicity only the `preprocess.py` summarizes all this functionality into one script.