if [ $# -ne 1 ]; then
  echo "Invalid syntax."
  echo "Syntax: simulate.sh <scene_name>"
else
  scene_name=$1

  n_frames=3
  n_objects=2
  img_size=64
  simulation_time=3

  #n_frames=200
  #n_objects=20
  #img_size=640
  #simulation_time=5

  if [ $scene_name == "robo_gears" ]; then
  	obj_scale=0.001
  elif [ $scene_name == "engine_parts" ]; then
  	obj_scale=0.01
  else
    	obj_scale=1.0
  fi

  
  #rm -fr output/simulation/$scene_name/*
  
  #if docker run --rm --interactive \
  #    --user $(id -u):$(id -g) \
  #    --volume "$PWD:/kubric" \
  #    kubricdockerhub/kubruntu \
  #    python3 simulate.py -s $scene_name --framerate $framerate --n_frames $n_frames --n_objects $n_objects ; then

    if 	(docker run --rm --interactive \
	 --user $(id -u):$(id -g) \
	 --volume "$PWD:/kubric" \
	 --gpus all \
	 --env KUBRIC_USE_GPU=1 \
	 kubricdockerhub/kubruntu \
	 python3 simulate.py -s $scene_name --simulation_time $simulation_time \
    --n_frames $n_frames --n_objects $n_objects --size $img_size --scale $obj_scale \
    ); then

    echo "Simulation done"
    
    #./make_gifs.sh $scene_name
    exit 0
  else
    echo "Failed"
    exit 1
  fi
fi
