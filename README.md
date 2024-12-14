# Cell Counting Made Easy: Leveraging Computer Vision for Microscopy Image Analysis 

We propose the use of object detection as an approach to count cells present in a microscopy image. To evaluate its accuracy, we used a dataset of images from the IDCIA v1 set for annotation and training and then images from the IDCIA v2 set for further testing.


**Code Contribution:**

- **Sadegh:** Contributed to the following files: `Ablation Study.ipynb`, `ACP5per.py`, `ADC Ablation Study.ipynb`, `ADC & Visualization.ipynb`, `filters.ipynb`, `MAE.py`, and `Training.ipynb`. Sadegh also completed part of the data annotation.

- **Nazifa:** Developed the `data visualization.ipynb` code and was one of the main annotators.


- **Rayan:** Served as the primary data annotator.

- **Thao:** She was more active in the other aspects of the project.



<img width="733" alt="Screenshot 2024-12-12 at 8 03 04 PM" src="https://github.com/user-attachments/assets/36a6de32-661c-4b40-ae81-c76418a7aea9" />


## You can download the datasets by running the following commands in the terminal.

## Dataset After Augmentation
```
curl -L "https://app.roboflow.com/ds/R5Q4XpbEbS?key=ZqK55vOxVM" > 
roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## Dataset Befote Augmentation
```
curl -L "https://app.roboflow.com/ds/RMavzafKfe?key=vR4b3yUfnP" > 
roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

Code Running Tip:

You can execute the codes on Pronto step by step to achieve the same results as we do. The only changes needed are the paths to the datasets and YAML files.

Code References:

Training.ipynb: https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb#scrollTo=tdSMcABDNKW-
