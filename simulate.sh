rm -f output/simulation/*

docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 simulate.py
    
echo "Simulation done, creating gif"

convert -delay 8 -loop 0 output/simulation/rgba_*.png output/simulator.gif
