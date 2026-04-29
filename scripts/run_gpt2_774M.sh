# GPT-2 (774M) repro on FineWeb
# 774M parameter model on ~150B tokens
# => 6 * 774e6 * 150e9 = 6.966e20 ~= 7e20 capability model (10X 350M)
# => 286,102 steps on 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~1.7s/iter
# => training time 286,102 steps * 1.7s = 135 hours ~= 5.6 days ~= $2000 (10X 124M)

make train_gpt2cu USE_CUDNN=0 # SDPA FP16/BF16 forward is not supported on cuDNN 9.10.0/9.10.1 due to known bugs. Please consider upgrading to 9.10.2 or newer.
out_dir="log_gpt2_774M"
done_file="$out_dir/DONE_00286102"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/fineweb.py --version 100B to prepro data
    # run python dev/data/hellaswag.py to prepro hellaswag eval
    mpirun -np 1 --bind-to none ./train_gpt2cu \
                -i "dev/data/minipile/m000/minipile_train_*.bin" \
                -j "dev/data/minipile/m000/minipile_val_*.bin" \
                -o $out_dir \
                -v 250 -s 500 -g 144 \
                -h 0 \
                -b 16 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.00025 \
                -q 0.0 \
                -u 700 \
                -n 4000 \
                -x 2810 \
                -y 1 \
                -e "d36"

    sleep 1
done
