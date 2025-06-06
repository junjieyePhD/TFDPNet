export CUDA_VISIBLE_DEVICES=0

model_name=TFDPNet

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_96_96_$pred_len \
      --model $model_name \
      --data traffic \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --factor 3 \
      --enc_in 862 \
      --des 'Exp' \
      --d_model 512 \
      --batch_size 16 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --n_heads 8
done
