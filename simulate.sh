
if [ $# -ne 1 ]; then
  echo "Invalid syntax."
  echo "Syntax: simulate.sh <scene_name>"
else
  scene_name=$1
  
  rm -f output/simulation/$scene_name/*
  
  if docker run --rm --interactive \
      --user $(id -u):$(id -g) \
      --volume "$PWD:/kubric" \
      kubricdockerhub/kubruntu \
      python3 simulate.py; then
      
    echo "Simulation done, creating gif"
    
    convert -delay 1 -loop 0 output/simulation/$scene_name/rgba_*.png output/simulator.gif
    
    firefox output/simulation/$scene_name/simulator.gif
    exit 0
  else
    echo "Failed"
    exit 1
  fi
fi
