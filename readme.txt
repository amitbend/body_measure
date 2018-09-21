set up environment

    1. clone the code
        git clone https://github.com/khanhha/body_measure.git

    2. build OpenPose
        1. cd to the folder body_measure

        2. clone OpenPose to the same folder
            git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

        3. follow this instruction to build OpenPose on  Linux 
            https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md
            Note: 
                I recommend not building OpenPose inside an Anaconda environemnt.
                I tried to do it for a few days but no luck. 
                Even I could build it successully, there're still run-time error due to library conflict.
                OpenPose should be built using native Python on OS and other OS libraries.

    3. create anaconda environment
        conda create -n body python=3.5
        conda activate body
        conda install -c conda-forge opencv
        conda install tensorflow-gpu shapely matplotlib pillow

run the code
    1. activate conda environment
        cd ./body_measure/src
        conda activate body

    2. extract pose: this program uses OpenPose to calculate pose and output pose data
       to the ouput folder
        run: pose_extract.py -i ../data/images -o ../data/pose

    3. extract silhouette: this program first downloads Deeplab model and then extract silhouette.
        The deeplab silhouete is then refined using local grab-cut and pose information
        
        run: silhouette.py -i ../data/images/ -p ../data/pose/ -o ../data/silhouette/

    4. extract body slices and measurement: this program uses silhouette and pose information to calculate body slices.
        For calculating measuremnt, height need to be passed in. 
        Unfortunately, I have't supprot passing in height parameter yet.
        
        run:  body_measure.py -i ../data/images -s ../data/silhouette -p ../data/pose -o ../data/measurement





