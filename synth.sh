dir="config/JVS-VCTK_langemb_configs/JVS-VCTK_"
configs=(0.01 0.001 0.01_woge2e 0.001_woge2e 0.5 0.5_woge2e 0.25 0.25_woge2e)

for config in ${configs[@]}; do
    echo $dir$config
    python synthesize.py -r 50000 -l en -t "I want to twist all things to my side, all realities" -s 9 -c $dir$config
done