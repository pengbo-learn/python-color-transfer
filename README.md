# Color Transfer in Python

Three methods of color transfer implemented in Python.


## Output Examples

| Input image  | Reference image  | Mean std transfer | Lab mean transfer | Pdf transfer + Regrain |
|--------------|------------|-------------------|-------------------|------------------------|
| ![example1_input](https://i.imgur.com/slFs7uz.jpeg) | ![example1_ref](https://i.imgur.com/CbcmZcW.png) | ![example1_mt](https://i.imgur.com/6NW8cgf.jpeg)         | ![example1_lt](https://i.imgur.com/M9iBNdJ.jpeg)         | ![example1_pdf-reg](https://i.imgur.com/4RUpleJ.jpeg)             |
| ![example2_input](https://i.imgur.com/3f92VzO.jpeg) | ![example2_ref](https://i.imgur.com/FE6fSiG.jpeg) | ![example2_mt](https://i.imgur.com/ssmM63F.jpeg)         | ![example2_lt](https://i.imgur.com/KXrFWbh.jpeg)         | ![example2_pdf-reg](https://i.imgur.com/MrslCTI.jpeg)             |
| ![example1_input](https://i.imgur.com/PHtfrPk.png) | ![example1_ref](https://i.imgur.com/LULa5k0.png) | ![example1_mt](https://i.imgur.com/RAYarUL.jpeg)         | ![example1_lt](https://i.imgur.com/ueoePsi.jpeg)         | ![example1_pdf-reg](https://i.imgur.com/rYHJW47.jpeg)             |
| ![example1_input](https://i.imgur.com/xCFLWda.png) | ![example1_ref](https://i.imgur.com/HZsiqyQ.jpeg) | ![example1_mt](https://i.imgur.com/jxeidOD.jpeg)         | ![example1_lt](https://i.imgur.com/GIUz83F.jpeg)         | ![example1_pdf-reg](https://i.imgur.com/faqeIdT.jpeg)             |

## Methods

Let input image be $I$, reference image be $R$ and output image be $O$.
Let $f{I}(r, g, b)$, $f{R}(r, g, b)$ be probability density functions of $I$ and $R$'s rgb values. 

- **Mean std transfer**

    $$O = (I - mean(I)) / std(I) \* std(R) + mean(R).$$

- **Lab mean transfer**[^1]

    $$I' = rgb2lab(I)$$
    $$R' = rgb2lab(R)$$
    $$O' = (I' - mean(I')) / std(I') \* std(R') + mean(R')$$
    $$O = lab2rgb(O')$$

- **Pdf transfer**[^2]

    $O = t(I)$, where $t: R^3-> R^3$ is a continous mapping so that $f{t(I)}(r, g, b) = f{R}(r, g, b)$. 


## Requirements
- 🐍 [python>=3.6](https://www.python.org/downloads/)


## Installation

### From PyPi

```bash
pip install python-color-transfer
```

### From source

```bash
git clone https://github.com/pengbo-learn/python-color-transfer.git
cd python-color-transfer

pip install -r requirements.txt
```

## Demo

- To replicate the results in [Output Examples](<#output-examples> "Output Examples"), run:

```bash
python demo.py 
```

<details>
  <summary>Output</summary>

```
demo_images/house.jpeg: 512x768x3
demo_images/hats.png: 512x768x3
Pdf transfer time: 1.47s
Regrain time: 1.16s
Mean std transfer time: 0.09s
Lab Mean std transfer time: 0.09s
Saved to demo_images/house_display.png

demo_images/fallingwater.png: 727x483x3
demo_images/autumn.jpg: 727x1000x3
Pdf transfer time: 1.87s
Regrain time: 0.87s
Mean std transfer time: 0.12s
Lab Mean std transfer time: 0.11s
Saved to demo_images/fallingwater_display.png

demo_images/tower.jpeg: 743x1280x3
demo_images/sunset.jpg: 743x1114x3
Pdf transfer time: 2.95s
Regrain time: 2.83s
Mean std transfer time: 0.23s
Lab Mean std transfer time: 0.21s
Saved to demo_images/tower_display.png
  
demo_images/scotland_house.png: 361x481x3
demo_images/scotland_plain.png: 361x481x3
Pdf transfer time: 0.67s
Regrain time: 0.49s
Mean std transfer time: 0.04s
Lab Mean std transfer time: 0.22s
Saved to demo_images/scotland_display.png
```

</details>


## Usage

```python
from pathlib import Path

import cv2
from python_color_transfer.color_transfer import ColorTransfer

# Using demo images
input_image = 'demo_images/house.jpeg'
ref_image = 'demo_images/hats.png'

# input image and reference image
img_arr_in = cv2.imread(input_image)
img_arr_ref = cv2.imread(ref_image)

# Initialize the class
PT = ColorTransfer()

# Pdf transfer
img_arr_pdf_reg = PT.pdf_tranfer(img_arr_in=img_arr_in,
                             img_arr_ref=img_arr_ref,
                             regrain=True)
# Mean std transfer
img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in,
                                  img_arr_ref=img_arr_ref)
# Lab mean transfer
img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)

# Save the example results
img_name = Path(input_image).stem
for method, img in [('pdf-reg', img_arr_pdf_reg), ('mt', img_arr_mt),
                   ('lt', img_arr_lt)]:
    cv2.imwrite(f'{img_name}_{method}.jpg', img)
```


[^1]: Lab mean transfer: [Color Transfer between Images](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf) by Erik Reinhard, Michael Ashikhmin, Bruce Gooch and Peter Shirley.\
    [Open source's python implementation](https://github.com/chia56028/Color-Transfer-between-Images)

[^2]: Pdf transfer: [Automated colour grading using colour distribution transfer](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694&rep=rep1&type=pdf) by F. Pitie , A. Kokaram and R. Dahyot.\
    [Author's matlab implementation](https://github.com/frcs/colour-transfer)
