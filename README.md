# Color Transfer in Python

A python implementation of [*Automated colour grading using colour distribution transfer*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.7694&rep=rep1&type=pdf) by F. Pitie , A. Kokaram and R. Dahyot.

# Examples
input img - reference img - transferred img
![img](imgs/house_display.png)
![img](imgs/scotland_display.png)

# Run
```bash
# python color_transfer.py 
/root/python_color_transfer/imgs/scotland_house.png: 361x481x3
/root/python_color_transfer/imgs/scotland_plain.png: 361x481x3
pdf transfer time: 0.66s
regrain time: 0.51s
save to /root/python_color_transfer/imgs/scotland_display.png
/root/python_color_transfer/imgs/house.jpeg: 512x768x3
/root/python_color_transfer/imgs/hats.png: 512x768x3
pdf transfer time: 1.41s
regrain time: 1.12s
save to /root/python_color_transfer/imgs/house_display.png
```

# References
[Author's matlab implementation](https://github.com/frcs/colour-transfer)
