dir_path="config/JVS-VCTK_langemb_configs/JVS-VCTK_1*"
dirs=`find $dir_path -maxdepth 0 -type d`

for dir in ${dirs[@]}; do
    echo $dir
done

echo -e "\n\n"

for dir in ${dirs[@]}; do
    echo $dir
    python train.py -c $dir --use_clf --checkpoint output/ckpt/JVS-VCTK_pretrain/20000.pth.tar --corpus JVS VCTK
    echo -e "\n train done! \n"
    python synthesize.py -r 50000 -l en -t "I want to twist all things to my side, all realities." -s 110 -c $dir
    python synthesize.py -r 50000 -l en -t "I want to twist all things to my side, all realities" -s 10 -c $dir
    python synthesize.py -r 50000 -l ja --use_accent -t "あらゆる現実をすべて自分の方へ捻じ曲げていきたい" -s 10 -c $dir
    python synthesize.py -r 50000 -l ja --use_accent -t "あらゆる現実をすべて自分の方へ捻じ曲げていきたい。" -s 110 -c $dir
    echo $dir
done