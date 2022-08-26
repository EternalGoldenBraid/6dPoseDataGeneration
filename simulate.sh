
if [ $# -ne 1 ]; then
  echo "Invalid syntax."
  echo "Syntax: simulate.sh <scene_name>"
else
  scene_name=$1
  framerate=6
  #n_frames=60
  n_frames=2
  n_objects=6
  
  rm -f output/simulation/$scene_name/*
  
  if docker run --rm --interactive \
      --user $(id -u):$(id -g) \
      --volume "$PWD:/kubric" \
      --gpus all \
      --env KUBRIC_USE_GPU=1 \
      kubricdockerhub/kubruntu \
      python3 simulate.py -s $scene_name --framerate $framerate --n_frames $n_frames --n_objects $n_objects ; then
      
    echo "Simulation done, creating gif"
    
    convert -delay 1 -loop 0 output/simulation/$scene_name/rgba_*.png output/simulation/$scene_name/simulator.gif
    
    firefox output/simulation/$scene_name/simulator.gif
    exit 0
  else
    echo "Failed"
    exit 1
  fi
fi
