# Segment-and-Track-Human-Using-Wireless-Signal
Human segmentation and tracking studied in [11785 - Introduction to Deep Learning] of Carnegie Mellon University in 2022 spring semester.

## Preparation 
### 1. Create the environment.     
```
$ cd ~/Segment-and-Track-Human-Using-Wireless-Signal
$ sh environment.sh
```     

### 2. Place the data and label in the path below.
```
Segment-and-Track-Human-Using-Wireless-Signal/    
└── dataset/
    └── data
          └── s1p0_1
          └── s1p0_2
          └── s1p0_3
                .
                .
          └── s1p2_9
          └── s1p2_10
    
    └── label
          └── s1p0_1
          └── s1p0_2
          └── s1p0_3
                .
                .
          └── s1p2_9
          └── s1p2_10
```   
       
### 3. Run main.sh for train
```
$ cd ~/Segment-and-Track-Human-Using-Wireless-Signal
$ sh train.sh
```       
            
### 4. Run test.sh for test   
```
$ cd ~/Segment-and-Track-Human-Using-Wireless-Signal
$ sh test.sh
```  
