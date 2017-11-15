# OpenFace 
*Free and open source face recognition with deep neural networks.*

OpenFace is a Python and Torch implementation of face recognition with deep neural networks and is based on the CVPR 2015
paper FaceNet: A Unified Embedding for Face Recognition and Clustering by Florian Schroff, Dmitry Kalenichenko, and James
Philbin at Google. Torch allows the network to be executed on a CPU or with CUDA.

OpenFace can be programmed in more languages (`C++, Python...`)

## The principle of functioning

1. Detect faces with a pre-trained models from `dlib` or `OpenCV`.
 
2. Transform the face for the neural network. This repository uses dlib's real-time pose 
   estimation with OpenCV's affine transformation to try to make the eyes and bottom lip appear in the same location on each image..
 
3. Use a deep neural network to represent (or embed) the face on a 128-dimensional unit hypersphere. The embedding is a 
   generic representation for anybody's face. Unlike other face representations, this embedding has the nice property that a 
   larger distance between two face embeddings means that the faces are likely not of the same person. This property makes 
   clustering, similarity detection, and classification 
   tasks easier than other face recognition techniques where the Euclidean distance between features is not meaningful.
   
4. Apply your favorite clustering or classification techniques to the features to complete your recognition task. 
   See below for our examples for classification and similarity detection, including an online web demo.
   
   ![functioning](https://raw.githubusercontent.com/cmusatyalab/openface/master/images/summary.jpg)

## Process

### Advanced Ubuntu installation

For Unix based systems and different compilers, I included Cmake files for cross-platform and cross-IDE support.

This code has been tested on Ubuntu 14.04.1 with GCC, and on 15.10 with Clang 3.7.1.

You can also run the install.sh script for installing on Ubuntu 16.04 (it combines the following steps into one script)




### Dependency installation

This requires cmake, OpenCV 3.1.0 (or newer), tbb and boost.

To acquire all of the dependencies follow the instructions pertaining to your Operating System:


End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc


