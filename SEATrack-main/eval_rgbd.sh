cd Depthtrack_workspace
vot evaluate --workspace ./ rgbd
vot analysis --nocache --name rgbd
cd ..
cd VOT22RGBD_workspace
vot evaluate --workspace ./ rgbd
vot analysis --nocache --name rgbd
cd ..