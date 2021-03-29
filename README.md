#A-Eye - AI that helps healthcare professionals diagnose eye disease

![A-Eye Application](https://www.tdumartin.com/img/A-Eye_WebDesign.png)

A-Eye is an AI that helps healthcare professionals diagnose eye disease based on a fundus image.
To carry out this project, we used a **dataset of 6,000 patients comprising 12,000 fundus images** of the eye labeled by doctors.

A-Eye is **3 deep learning models (Convolutional Neural Network)** which follow one another to achieve the most accurate results possible.

👉 The first model detects whether the image matches a fundus image.<br>
The model was trained on a database of **60,000 images classified into 10 categories**.
![A-Eye Application](https://www.tdumartin.com/img/A-Eye_Webdesign_W.png)


👉 The second model detects whether the eye is healthy or sick.<br>
For this second model, we used a transfer learning method, **InceptionV3**, a pre-trained model with more than 23 million parameters.
![A-Eye Application](https://www.tdumartin.com/img/A-Eye_Webdesign_N.png)

👉 Finally, the third model detects the disease. (6 possible categories: Cataract / Glaucoma / Macular Degeneration / Myopia / Hypertension / Other)
We used the **VGG16**, a pre-trained model with over 150 million parameters.
![A-Eye Application](https://www.tdumartin.com/img/A-Eye_Webdesign_D.png)


Tools:
Python / Google Cloud Platform / Docker / Streamlit / Heroku / etc ...
