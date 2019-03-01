Description:
This program predicts whether a given comment contains derogatory or abusive content or not.
Two machine learning algorithms: support vector machines and multinomial naive bayes are
used for this purpose.

Prerequisites:
The program is written in Python 3.6 . The following libraries are required to run this program :
1. sklearn
2. numpy
3. pandas
4. os
5. re
6. csv

Installation:
1. Pip can be installed by using the following command:
sudo easy_install pip
2. Scikit-learn can be installed by using the following command:
sudo pip install -U numpy scipy scikit-learn
3. Pandas can be installed by using the following command:
pip install pandas
Other libraries can be installed by using the pip command in a similar way shown above.

Instructions to run:
1. The program requires the dataset to be present in the directory which can be found here .
2. The files needed are train.csv, impermium_verification_labels.csv and
test_with_solutions.csv (Note: Remove the column Usage from the file
test_with_solutions.csv)
3. To increase the number of training instances, we have merged the files train.csv and
impermium_verification_labels.csv. train.csv can also be used individually.
4. Create two directories where the python code is located. Name them data and
cleaned_data/data and store the files from the given link in the directory: data.
5. Run the program by using the following command: python abusive_content_detection.py

Authors:
The authors for this program are:
Prajakta Gaydhani(pag3862), Virtee Parekh(vvp2639), Vaibhav Nagda(vjn4006).
