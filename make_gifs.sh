#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Invalid syntax."
  echo "Syntax: make_giffs.sh <scene_name>"
else

  scene_name=$1

  data_path=output/simulation/$scene_name
  data_run_ids=$(ls $data_path)
  echo "Creating giff for scene $scene_name"
  echo "Found ids:"
  echo $data_run_ids
  echo 

  found_paths=$(find $data_path -nowarn -name *.gif);
  echo "Ignoring gif(s) at"
  echo "$found_paths"
  echo

  # https://stackoverflow.com/questions/3578584/bash-how-to-delete-elements-from-an-array-based-on-a-pattern
  shopt -s extglob
  for found in $found_paths;
  do
    id=$(echo $found | awk -F " +|/" '{print $(NF-1)}');
    data_run_ids="${data_run_ids//$id}"
  done

  for id in $data_run_ids;
  do
  
    echo "Converting from: $(ls $data_path/$id/rgb/rgb_*.png)"
    echo "To: $data_path/$id/rgb.gif"
    echo
    convert -delay 100 -loop 0 $data_path/$id/rgb/rgb_*.png $data_path/$id/rgb.gif

  done
fi
