# Introduction
The World Canine Organization currently recognizes 339 breeds of dogs. One might think that identifying dog breeds is a simple task, but some breeds look so similar that can become difficult for humans to classify. For Example, in the Image below we can observe two different breeds that look almost the same. 

![alt text](https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F47%2F2021%2F06%2F15%2FBoston-Terrier-vs-French-Bulldog-02-2-2000.jpg)

The image on the left is a Boston Terrier and the Image on the right is a French Bulldog. As you can observe it is not that easy for humans to classify all dog breeds correctly. What about a computer, can a computer classify dog breeds better than a human? In this project I build a dog image classification model using InceptionV3 with 88% accuracy and I was able to deploy the model online using Streamlit. 

## :ledger: Index

- [Introduction](#Introduction)
- [Usage](#zap-Usage)
- [File Structure](#file_folder-file-structure) 
- [Contribution](#fire-contribution)
- [Conclusions](#dog-Conclusion)

## :zap: Usage
- Vist the website by clicking the link below and Upload a Dog Image. 
- [Website](https://carlos-lesser-dog-app-dog-app-38wto4.streamlitapp.com/)
- The Notebook file is a walkthrough my entire project. In the Notebook you will find all the Code and a detail explanation of the project. 

##  :file_folder: File Structure
Add a file structure here with the basic details about files, below is an example.

```

├── Notebook
│   ├── Dog Breed Image Classification Notebook.md
│   ├── output_12_0.png
│   ├── output_18_0.png
│   ├── output_23_0.png
│   ├── output_25_0.png
│   ├── output_32_0.png
│   ├── output_36_0.png   
│   ├── output_56_0.png
│   ├── output_62_0.png   
│   ├── output_62_2.png
│   ├── output_62_4.png
│   ├── output_62_6.png
│   ├── output_62_8.png
│   ├── output_64_0.png
├── .gitatributtes
├── Procfile.txt
├── README.md
├── dog_app.py
├── model.h5
├── model.json
├── requirements.txt
└── setups.sh

```

 ##  :fire: Contribution

 Your contributions are always welcome and appreciated.

 
##  :dog: Conclusion
The model performs well with 88% accuracy on the test. I was expecting to reach 70% accuracy on the model, and I was able to achieve that. The next time I do a classification problem, I would focus more on precision, recall and F1 scores.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>123.000000</td>
      <td>123.000000</td>
      <td>123.000000</td>
      <td>123.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.891019</td>
      <td>0.883867</td>
      <td>0.882351</td>
      <td>52.470619</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.103806</td>
      <td>0.124467</td>
      <td>0.100322</td>
      <td>270.915787</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500000</td>
      <td>0.450000</td>
      <td>0.500000</td>
      <td>0.886099</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.846053</td>
      <td>0.866667</td>
      <td>0.852814</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.923077</td>
      <td>0.933333</td>
      <td>0.916667</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.952381</td>
      <td>0.954545</td>
      <td>0.945906</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2151.000000</td>
    </tr>
  </tbody>
</table>
</div>

This project was fun, and it gave me a better understanding of CNN models and transfer learning. This knowledge can be applied to different industries such as manufacturing (for defect detection), medicine (to detect anomalies in organs or tissues), agriculture (to identify which plans need water or to spot plagues on the crops), commerce (to identify similar products). The next step would be to make an image recognition model to identify if the image is of a dog.


