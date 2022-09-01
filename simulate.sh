
if [ $# -ne 1 ]; then
  echo "Invalid syntax."
  echo "Syntax: simulate.sh <scene_name>"
else
  scene_name=$1

  framerate=6
  n_frames=2
  n_objects=2

  #framerate=30
  #n_frames=30
  #n_objects=2
  
  rm -fr output/simulation/$scene_name/*
  
  #if docker run --rm --interactive \
  #    --user $(id -u):$(id -g) \
  #    --volume "$PWD:/kubric" \
  #    kubricdockerhub/kubruntu \
  #    python3 simulate.py -s $scene_name --framerate $framerate --n_frames $n_frames --n_objects $n_objects ; then

#  background_name=$( \
#	docker run --rm --interactive \
#	 --user $(id -u):$(id -g) \
#	 --volume "$PWD:/kubric" \
#	 --gpus all \
#	 --env KUBRIC_USE_GPU=1 \
#	 kubricdockerhub/kubruntu \
#	 python3 simulate.py -s $scene_name --framerate $framerate --n_frames $n_frames --n_objects $n_objects )

  background_name=true
  if $background_name; then 
	docker run --rm --interactive \
	 --user $(id -u):$(id -g) \
	 --volume "$PWD:/kubric" \
	 --gpus all \
	 --env KUBRIC_USE_GPU=1 \
	 kubricdockerhub/kubruntu \
	 python3 simulate.py -s $scene_name --framerate $framerate --n_frames $n_frames --n_objects $n_objects
      
	echo "Background name: $background_name"
	exit 0
    echo "Simulation done, creating gif"
    
    convert -delay 1 -loop 0 output/simulation/$scene_name/$background_name/rgba_*.png output/simulation/$scene_name/$(background_name).gif
    
    #firefox output/simulation/$scene_name/simulator.gif

    #ip_addr=130.230.19.5
    #rsync output/simulation/$scene_name/simulator.gif \
    #        nicklas@$ip_addr:Projects/6dPoseDataGeneration/output/simulation/$scene_name/simulator.gif
    #rsync output/simulation/$scene_name/simulator.blend \
#	    nicklas@$id_addr:Projects/6dPoseDataGeneration/output/simulation/$scene_name/simulator.blend
    exit 0
  else
    echo "Failed"
    echo $background_name
    exit 1
  fi
fi
