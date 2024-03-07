# russian-nouns-final
This is the code base for my bachelors thesis on "Investigation of the Role of Semantic Classes for Shift Vectors in Russian Language"

Installation:
1. Make sure you have a python interpreter, pip and the jupyter package installed on your system.
2. To install the needed libraries and depencies. You can do so by executing pip install -r requirements.txt
3. Run the Jupyter notebook russian_noun_analysis.ipynb to download the model and generate all outputs and plots

Issue handling:
- To install fasttext on your system you might require additional dependencies like Visual Studio Build Tools on Windows. Please visit the official fasttext documentation page for more details on that.
- During execution of the jupyter notebook you may run into troubles downloading the fasttext model. If the download stops, just try again. If everything has worked successfully you should see a models folder in your project directory with the files cc.ru.300.bin and cc.ru.300.bin.gz
- Sometimes you might end up with a models folder inside the models folder. You can delete the second models folder that is inside the other one.
- The plots are generated with timestamp file names. If you execute the jupyter notebook several times you might end up with multiple duplicate plots. You can delete the plots if you like before running the jupyter notebook again.