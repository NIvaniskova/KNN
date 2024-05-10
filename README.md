# Convolutional Neural Networks
## Identification of people by face

### Style transfer
To create the dataset for our needs a style transfer model needed to be used to generate paint, drawing, sculpture-like 
images. Style transfer was done using a TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

### Preprocessing dataset for model training
The names of the files and directories of the dataset need to follow a convention of AdaFace (presented by FaceNet) thus
dedicated scripts and instructions are provided in the `preprocessing` directory.