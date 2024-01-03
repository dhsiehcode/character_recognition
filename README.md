# Character Recognition

### About

This program recognizes handwritten 0-9 and A-Z characters. It uses a ResNet structure with 50 layers.
Both files perform the same.

### Data Source

(https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) and (https://www.kaggle.com/competitions/digit-recognizer/data)

### Result

Classes are labelled as:
- Labels 0 - 9 are 0 - 9 in that order
- Labels 10 - 26 are A-Z in that order (i.e. A is class 10)

![](https://github.com/dhsiehcode/character_recognition/blob/272b4ee6f096977664098650de599c6ae432e770/plots/roc_plot.png)
![](https://github.com/dhsiehcode/character_recognition/blob/272b4ee6f096977664098650de599c6ae432e770/plots/multi_accs.png)
![](https://github.com/dhsiehcode/character_recognition/blob/272b4ee6f096977664098650de599c6ae432e770/plots/confmat.png)

### Next Step


Incorporate a-z
