#export CUDA_VISIBLE_DEVICES=1


model_name=TFDPNet


for pred_len in 96 192 336 720
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id "ETTm1_96_96_$pred_len" \
        --model "$model_name" \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 7 \
        --des 'Exp' \
        --d_model 128 \
        --batch_size 32 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --n_heads 2
done

