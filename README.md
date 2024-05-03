# K-Means Clustering for Image Segmentation

This repository contains code for performing K-Means clustering on an RGB image for the purpose of image segmentation. The K-Means algorithm partitions the pixels of the image into K clusters by minimizing the sum of squared distances between each pixel and its assigned cluster center.

## Requirements

- [Python](https://www.python.org/) and [pip](https://pip.pypa.io/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## Installation




1. Clone the repository:

    ```bash
    git clone https://github.com/ugendar07/K-Means.git
    ```

 
   

## Usage

1. Place your image file in the `images` directory.
2. Open the `utils.py` file and modify the `image_path` variable to point to your image file.
3. Run the script:

    ```bash
    main.py
    ```



    

4. The segmented images will be saved in the `results` directory.

## Parameters

You can adjust the number of clusters (`k`) by modifying the `k_values` list in the script. By default, the script will perform K-Means clustering for `k = [2, 5, 10, 20, 50]`.

## Results

- The results of the segmentation will be saved in the `results` directory. Each segmented image will be named according to the number of clusters used.
- Plot for K vs MSE also saved in the `results` directory





## Contact
For questions or inquiries, please contact [ugendar](mailto:ugendar07@gmail.com) .

  

