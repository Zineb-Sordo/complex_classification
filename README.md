# Complex-valued classification of Knee pathologies 

The implementation of this code uses the paper "Importance of Alternate Signal Representation in Automatic MR Image Diagnosis" [1] as baseline and creates a complex-valued deep network for the classification of knee pathologies. 

## Abstract 

Magnetic Resonance imaging (MRI) is a commonly used medical imaging method that acquires MR signals represented by k-space which contains complex-valued frequency space measurements. The visually readable images are then obtained using the inverse Fourier Transform of the magnitude of these complex measurements, and analyzed by the physi- cians to establish a medical diagnosis. There have been numerous studies on the automation of medical diagnosis using magnitude-based image datasets as input to Deep Learning mod- els, which demonstrated very encouraging results. However, in doing so, the images chosen discard a part of the information contained in the complex-valued MR measurements no- tably in the phase of the signal. Some work has also highlighted the improvement of such results when using real-valued alternate signal representations with real-valued deep net- works. Therefore, this paper explores a novel approach that takes into account the phase information by using a raw complex-valued k-space dataset with a complex-valued net- work and focuses on comparing the classification performance of real-valued deep neural networks with real-valued alternate signal representations.

## Installations

- Requirements: to install the required dependencies, run the following:  
```bash
$ pip install -r requirements.txt
```
For data generation and processing, start by following the same steps as in paper [1] [GitHub](https://github.com/anonycodes/MRI):
* __Data Generation__ 

The fastMRI can be downloaded from [this link](https://fastmri.med.nyu.edu) and the annotations can be found at [this link](https://github.com/microsoft/fastmri-plus/tree/main/Annotations). The original fastMRI data contains volume level slices so generate the Slice level processed data after updating the required paths in the file:
 ```
$ cd data_generation
$ python knee_singlecoil.py
```
* Generate the train, validation, and test splits
```
$ cd data_generation
$ python generate_knee_metadata.py
```

To standardize the data on the full training set, after the data generation process, run the following: 
```

$ python scale_complex_data.py
```

To train the model, run the script (after adding the appropriate argparse arguments): 
```

$ python model_training.py
```

This model can be trained using scaled and unscaled data, 4 different activation functions, the type of output after the final complex layer.

